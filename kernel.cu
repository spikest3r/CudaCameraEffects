#include <glad/glad.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <device_functions.h>
#include <GLFW/glfw3.h>
#include "cuda_gl_interop.h"
#include "shader.h"
#include <atomic>

#include "NvEncoder/NvEncoderCuda.h"
#include "NvEncoder/NvEncoderOutputInVidMemCuda.h"
#include "Logger.h"

#include "cuda.h"

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/avutil.h>
}

struct EncodeBuffer {
    unsigned char* frame;
    bool available;
};

std::atomic<int> writeIndex{ 0 };
std::atomic<int> readIndex{ 0 };
EncodeBuffer buffers[4];

uint8_t effects = 0b11111111;
bool encodeVideo = false;
bool running = true;
GLFWwindow* window;

int frame_index = 0;
cudaStream_t nvencStream;
AVStream* stream;
AVFormatContext* fmt_ctx = nullptr;
CUdeviceptr dFrame;
CUcontext cuContext = nullptr;

inline uint8_t getBit(uint8_t byte, uint8_t bit)
{
    return (byte >> bit) & 1;
}

inline void setBit(uint8_t& byte, uint8_t bit)
{
    byte ^= (1 << bit);
}

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

__global__ void RGB2NV12(unsigned char* frame, uint8_t* pFrame, int width, int height)
{
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

     int idx = (x + y * width) * 3;

     unsigned char R = frame[idx + 2];
     unsigned char G = frame[idx + 1];
     unsigned char B = frame[idx + 0];

     unsigned char Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16;
     unsigned char U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128;
     unsigned char V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128;

    // Y plane
     int yIndex = y * width + x;
    pFrame[yIndex] = Y;

    // UV plane (1 sample per 2x2 pixels)
    if ((x % 2 == 0) && (y % 2 == 0)) {
         int uvWidth = width / 2;
         int uvX = x / 2;
         int uvY = y / 2;
         int uvIndex = width * height + (uvY * uvWidth + uvX) * 2;
        pFrame[uvIndex + 0] = U; // U
        pFrame[uvIndex + 1] = V; // V
    }
}

float quadVertices[] = {
    // positions   // texcoords
    -1.f, -1.f,   0.f, 0.f,
     1.f, -1.f,   1.f, 0.f,
     1.f,  1.f,   1.f, 1.f,

    -1.f, -1.f,   0.f, 0.f,
     1.f,  1.f,   1.f, 1.f,
    -1.f,  1.f,   0.f, 1.f
};

const char* vertexCode = R"(#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;
out vec2 TexCoord;
void main() {
    TexCoord = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
})";

const char* fragmentCode = R"(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
void main() {
    FragColor = texture(uTexture, TexCoord);
})";


template <typename T>
__device__ T clamp(T val, T minVal, T maxVal) {
    return max(min(val, maxVal), minVal);
}

__global__ void __launch_bounds__(256, 6) blitGl(unsigned char* frame, cudaSurfaceObject_t surface, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= w || y >= h) return;

    unsigned char* p = frame + (x + y * w) * 3;

    uchar4 out;
    out.x = p[2]; // B
    out.y = p[1]; // G
    out.z = p[0]; // R
    out.w = 255;  // A

    surf2Dwrite(out, surface, x * 4, h - y - 1);
}

__global__ void mirrorX(unsigned char* in, unsigned char* out, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= w || y >= h) return;

    int idx = (x + y * w) * 3;
    int idxMirror = ((w - x - 1) + y * w) * 3;

    out[idx] = in[idxMirror];
    out[idx + 1] = in[idxMirror + 1];
    out[idx + 2] = in[idxMirror + 2];
}


__global__ void mosaic(unsigned char* in, unsigned char* out, int w, int h, int B) {
    __shared__ int sr, sg, sb, count;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * B + tx;
    int y = blockIdx.y * B + ty;

    if (tx == 0 && ty == 0) {
        sr = sg = sb = 0;
        int bw = min(B, max(0, w - blockIdx.x * B));
        int bh = min(B, max(0, h - blockIdx.y * B));
        count = bw * bh;
    }
    __syncthreads();

    if (x < w && y < h) {
        int idx = (y * w + x) * 3;
        atomicAdd(&sr, in[idx + 0]);
        atomicAdd(&sg, in[idx + 1]);
        atomicAdd(&sb, in[idx + 2]);
    }
    __syncthreads();
    
    int r = sr / count;
    int g = sg / count;
    int b = sb / count;

    if (x < w && y < h) {
        int outIdx = (y * w + x) * 3;
        out[outIdx + 0] = r;
        out[outIdx + 1] = g;
        out[outIdx + 2] = b;
    }
}


__global__ void negative(unsigned char* in, unsigned char* out, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= w || y >= h) return;

    int idx = (x + y * w) * 3;

    out[idx] = 255 - in[idx];
    out[idx + 1] = 255 - in[idx + 1];
    out[idx + 2] = 255 - in[idx + 2];
}

__constant__ constexpr float amplitude = 0.1f;   // radians, small twist per radius
__constant__ constexpr float frequency = 0.1f;   // controls number of sine waves outwards

__device__ __forceinline__ float fast_atan2(float y, float x) {
    float abs_x = __int_as_float(__float_as_int(x) & 0x7fffffff);
    float abs_y = __int_as_float(__float_as_int(y) & 0x7fffffff);

    float a = (abs_x < abs_y) ? abs_x : abs_y;
    float b = (abs_x > abs_y) ? abs_x : abs_y;

    float s = (b > 0.0f) ? a/b : 0.0f;

    float t = s * s;
    float r = ((-0.0464964749f * t + 0.15931422f) * t - 0.327622764f) * t * s + s;

    if (abs_y > abs_x) r = 1.57079633f - r;
    if (x < 0.0f) r = 3.14159265f - r;
    if (y < 0.0f) r = -r;

    return r;
}

__global__ void __launch_bounds__(256, 6)
spiral(unsigned char* in, unsigned char* out, int w, int h, float cx, float cy, float phase) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= w || y >= h) return;

    float u = (float)x - cx;
    float v = (float)y - cy;

    float r = __fsqrt_rn(u * u + v * v);
    u = fast_atan2(v, u) + 0.1f * __sinf(0.1f * r + phase);

    float nx = __fdividef(cx + r * __cosf(u), (float)(w - 1));
    float ny = __fdividef(cy + r * __sinf(u), (float)(h - 1));

    int xi = __float2int_rn(__saturatef(nx) * (float)(w - 1));
    int yi = __float2int_rn(__saturatef(ny) * (float)(h - 1));

    reinterpret_cast<uchar3*>(out)[y * w + x] = reinterpret_cast<uchar3*>(in)[yi * w + xi];
}

void pollKeys() {
    while (running) {
        for (int i = GLFW_KEY_1; i <= GLFW_KEY_8; i++) {
            int j = i - GLFW_KEY_1;
            if (glfwGetKey(window, i) == GLFW_PRESS) {
                setBit(effects, j);
                while (glfwGetKey(window, i) == GLFW_PRESS);
            }
        }

        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            encodeVideo = !encodeVideo;
            glfwSetWindowTitle(window, encodeVideo ? "Encoding video..." : "Camera");
            while (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
        }
    }
}

std::unique_ptr<NvEncoderCuda> pEnc;

void encodeThread() {
    cudaSetDevice(0);

    dim3 block(16, 16); // 32x32 threads per block
    dim3 grid(
        (1280 + block.x - 1) / block.x,  // ceil division
              (720 + block.y - 1) / block.y
    );

    while (running) {
        int idx = readIndex % 4;
        if (!buffers[idx].available) {
            RGB2NV12 << <grid, block, 0, nvencStream >> > (buffers[idx].frame, (uint8_t*)dFrame, 1280, 720);

            const NvEncInputFrame* encoderInputFrame = pEnc->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(
                cuContext,
                (uint8_t*)dFrame, 0, // device ptr
                (CUdeviceptr)encoderInputFrame->inputPtr,
                (int)encoderInputFrame->pitch,
                1280,
                720,
                CU_MEMORYTYPE_DEVICE, // GPU→GPU
                encoderInputFrame->bufferFormat,
                encoderInputFrame->chromaOffsets,
                encoderInputFrame->numChromaPlanes,
                false,
                nvencStream
            );

            std::vector<std::vector<uint8_t>> vPacket;
            pEnc->EncodeFrame(vPacket);
            for (auto& packet : vPacket) {
                AVPacket pkt;
                av_init_packet(&pkt);
                pkt.data = packet.data();
                pkt.size = packet.size();

                pkt.pts = av_rescale_q(frame_index, {1, 30}, stream->time_base);
                pkt.dts = pkt.pts;

                pkt.stream_index = stream->index;
                pkt.duration = av_rescale_q(1, {1, 30}, stream->time_base);

                av_interleaved_write_frame(fmt_ctx, &pkt);
                frame_index++;
            }

            // encode buffers[idx]
            buffers[idx].available = true;
            readIndex++;
        }
    }
}

int main() {
    cudaSetDevice(0);

    avformat_network_init();

    avformat_alloc_output_context2(&fmt_ctx, nullptr, "mp4", "camera.mp4");
    if(!fmt_ctx) {
        std::cerr << "Could not allocate format context\n";
        return -1;
    }

    stream = avformat_new_stream(fmt_ctx, nullptr);
    stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    stream->codecpar->codec_id = AV_CODEC_ID_H265;
    stream->codecpar->width = 1280;
    stream->codecpar->height = 720;
    stream->time_base = {1, 30};

    if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&fmt_ctx->pb, "camera.mp4", AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Could not open output file\n";
            return -1;
        }
    }

    avformat_write_header(fmt_ctx, nullptr);

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    if (!cap.isOpened()) {
        std::cerr << "Can't open camera!\n";
        return -1;
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(1280, 720, "Camera", nullptr, nullptr);
    if (!window) {
        cap.release();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        cap.release();
        return -2;
    }

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // position
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texcoord
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    size_t frameSize = 1280 * 720 * 3 * sizeof(unsigned char);

    ck(cuInit(0));
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, 0));
    ck(cuCtxCreate(&cuContext, 0, 0, cuDevice));

    pEnc = std::unique_ptr<NvEncoderCuda>(new NvEncoderCuda(cuContext, 1280, 720, NV_ENC_BUFFER_FORMAT_NV12));

    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;

    pEnc->CreateDefaultEncoderParams(&initializeParams, NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P3_GUID, NV_ENC_TUNING_INFO_LOW_LATENCY);
    pEnc->CreateEncoder(&initializeParams);

    int nFrameSize = pEnc->GetFrameSize();

    // Allocate GPU frame buffer
    ck(cuMemAlloc(&dFrame, nFrameSize));

    unsigned char* pinned_frame;
    cudaHostAlloc(&pinned_frame, frameSize, cudaHostAllocDefault);

    unsigned char* d1;
    cudaMalloc(&d1, frameSize);

    unsigned char* d2;
    cudaMalloc(&d2, frameSize);

    for (int i = 0; i < 4; i++) {
        cudaMalloc(&buffers[i].frame, frameSize);
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1280, 720, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    glBindTexture(GL_TEXTURE_2D, 0);

    Shader shader(vertexCode, fragmentCode);

    dim3 block(16, 16); // 32x32 threads per block
    dim3 grid(
        (1280 + block.x - 1) / block.x,  // ceil division
        (720 + block.y - 1) / block.y
    );

    cudaStream_t dataStream;
    cudaStream_t processStream;

    cudaStreamCreate(&dataStream);
    cudaStreamCreate(&processStream);
    cudaStreamCreate(&nvencStream);

    cudaEvent_t eventA;
    
    cudaEventCreate(&eventA);

    cudaArray_t cuArray;
    cudaGraphicsMapResources(1, &cudaResource, dataStream);
    cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);

    cv::Mat frame;

    std::thread thread(pollKeys);

    float spiralT = 0.0f;

    float oldTime = 0.0f;

    for(int i = 0; i < 4; i++) {
        buffers[i].available = true;
    }

    std::thread encThread(encodeThread);

    while (!glfwWindowShouldClose(window)) {
        // deltatime
        cudaGraphicsMapResources(1, &cudaResource, dataStream);
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - oldTime;
        oldTime = currentTime;

        spiralT += deltaTime * 10.0f;

        cap >> frame; // grab a frame
        unsigned char* dataPtr = frame.data;

        // cuda
        cudaMemcpyAsync(pinned_frame, dataPtr, frameSize, cudaMemcpyHostToDevice, dataStream);

        cudaMemcpyAsync(d1, pinned_frame, frameSize, cudaMemcpyDeviceToDevice, dataStream);

        if (getBit(effects, 0)) {
            dim3 block2(8, 8);
            dim3 grid2(
                (1280 + block2.x - 1) / block2.x,  // ceil division
                (720 + block2.y - 1) / block2.y
            );
            mosaic << <grid2, block2, 0, dataStream >> > (d1, d2, 1280, 720, 8);
            std::swap(d1, d2);
        }
        if (getBit(effects, 1)) {
            negative << <grid, block, 0, dataStream >> > (d1, d2, 1280, 720);
            std::swap(d1, d2);
        }
        if (getBit(effects, 2)) {
            mirrorX << <grid, block, 0, dataStream >> > (d1, d2, 1280, 720);
            std::swap(d1, d2);
        }
        if (getBit(effects, 3)) {
            spiral << <grid, block >> > (d1, d2, 1280, 720, 1280/2, 720/2, spiralT);
            std::swap(d1, d2);
        }

        cudaEventRecord(eventA, dataStream);

        cudaStreamWaitEvent(processStream, eventA);
        blitGl <<<grid, block, 0, processStream>>> (d1, surface, 1280, 720);

        if (encodeVideo) {
            cudaStreamWaitEvent(nvencStream, eventA);

            int idx = writeIndex % 4;
            if (buffers[idx].available) {
                cudaMemcpyAsync(buffers[idx].frame, d1, frameSize, cudaMemcpyDeviceToDevice, nvencStream);
                cudaEventRecord(eventA, nvencStream);
                buffers[idx].available = false;
                writeIndex++;
            }
        }

        cudaGraphicsUnmapResources(1, &cudaResource, dataStream);

        // render loop
        shader.use();
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glDrawArrays(GL_TRIANGLES, 0, 6); // 6 vertices, 2 triangles

        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    // std::vector<std::vector<uint8_t>> vPacket;
    // pEnc->EndEncode(vPacket);
    // for (auto& packet : vPacket) {
    //     AVPacket pkt;
    //     av_init_packet(&pkt);
    //     pkt.data = packet.data();
    //     pkt.size = packet.size();
    //     pkt.pts  = frame_index;
    //     pkt.dts  = frame_index;
    //     pkt.stream_index = stream->index;
    //
    //     av_interleaved_write_frame(fmt_ctx, &pkt);
    //     frame_index++;
    // }

    av_write_trailer(fmt_ctx);
    avio_closep(&fmt_ctx->pb);
    avformat_free_context(fmt_ctx);

    pEnc->DestroyEncoder();
    ck(cuMemFree(dFrame));
    ck(cuCtxDestroy(cuContext));

    running = false;
    if(encThread.joinable()) encThread.join();
    if(thread.joinable()) thread.join();

    cudaDestroySurfaceObject(surface);
    cudaFreeHost(pinned_frame);
    cudaFree(d1);
    cudaFree(d2);

    cap.release();
    return 0;
}
