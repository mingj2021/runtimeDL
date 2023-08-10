#include "general.h"
#include "buffers.h"
#include "yolo.h"
#include <iostream>
#include <fstream>
#include "ThreadPool.h"
#include <chrono>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <opencv2/core/utility.hpp>

using namespace algorithms;
using namespace std::chrono;
using namespace nvinfer1;
using namespace nvonnxparser;

#undef CHECK
#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)


void index2srt(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        std::cout << "nvinfer1::DataType::kFLOAT" << std::endl;
        break;
    case nvinfer1::DataType::kHALF:
        std::cout << "nvinfer1::DataType::kHALF" << std::endl;
        break;
    case nvinfer1::DataType::kINT8:
        std::cout << "nvinfer1::DataType::kINT8" << std::endl;
        break;
    case nvinfer1::DataType::kINT32:
        std::cout << "nvinfer1::DataType::kINT32" << std::endl;
        break;
    case nvinfer1::DataType::kBOOL:
        std::cout << "nvinfer1::DataType::kBOOL" << std::endl;
        break;
    case nvinfer1::DataType::kUINT8:
        std::cout << "nvinfer1::DataType::kUINT8" << std::endl;
        break;

    default:
        break;
    }
}

void dims2str(nvinfer1::Dims dims)
{
    std::string o_s("[");
    for (size_t i = 0; i < dims.nbDims; i++)
    {
        if (i > 0)
            o_s += ", ";
        o_s += std::to_string(dims.d[i]);
    }
    o_s += "]";
    std::cout << o_s << std::endl;
}

// To create a builder, you first must instantiate the ILogger interface
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

class InferBuffer
{
public:
    InferBuffer(std::string bufferName,std::shared_ptr<nvinfer1::ICudaEngine> &engine, cv::Mat im, bool is_seg = true, int width = 640, int height = 640):
        mBufferName(bufferName), mEngine(engine), frame(im),inp_width(width), inp_height(height),m_is_seg(is_seg)
    {
        names = read_names("/workspace/runtimeDL/data/coco.names");
        context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!context)
        {
            std::cerr << "create context error" << std::endl;
        }

        CHECK(cudaStreamCreate(&stream));
        CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
        CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

        for (int i = 0; i < mEngine->getNbBindings(); i++)
        {
            auto dims = mEngine->getBindingDimensions(i);
            auto tensor_name = mEngine->getBindingName(i);
            std::cout << "tensor_name: " << tensor_name << std::endl;
            dims2str(dims);
            nvinfer1::DataType type = mEngine->getBindingDataType(i);
            index2srt(type);
            int vecDim = mEngine->getBindingVectorizedDim(i);
            //std::cout << "vecDim:" << vecDim << std::endl;
            //if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
            //{
            //    int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
            //    std::cout << "scalarsPerVec" << scalarsPerVec << std::endl;
            //}
            auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{ 1 }, std::multiplies<int64_t>{});
            std::unique_ptr<algorithms::DeviceBuffer> device_buffer{ new algorithms::DeviceBuffer(vol, type) };
            mDeviceBindings.emplace_back(device_buffer->data());
            mInOut[tensor_name] = std::move(device_buffer);
        }
    }

    ~InferBuffer()
    {
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(end));
        CHECK(cudaStreamDestroy(stream));
    }
    int prepareInput()
    {
        steady_clock::time_point start_t = steady_clock::now();
        pad_info = algorithms::letterbox(frame, img, cv::Size(inp_width, inp_height));

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1.0 / 255);

        inputs =
            torch::from_blob(img.data, { img.rows, img.cols, img.channels() })
            .permute({ 2, 0, 1 }).contiguous().to(torch::kHalf);//
        // torch::Tensor x;
        // (x.data_ptr<at::Half>());
        auto ret = mInOut["images"]->host2device((void*)(inputs.data_ptr<at::Half>()), true, stream);
        steady_clock::time_point end_t = steady_clock::now();
        long long diff = (std::chrono::time_point_cast<milliseconds>(end_t) -
            std::chrono::time_point_cast<milliseconds>(start_t))
            .count();
        std::cout << mBufferName << " prepareInput costs time: " << diff << std::endl;
        return ret;
    }

    bool infer()
    {
        CHECK(cudaEventRecord(start, stream));
        auto ret = context->enqueueV2(mDeviceBindings.data(),stream, nullptr);
        return ret;
    }

    int verifyOutput()
    {
        //CHECK(cudaStreamSynchronize(stream));
        steady_clock::time_point start_t;
        float ms{ 0.0f };
        const float pad_w = pad_info[0];
        const float pad_h = pad_info[1];
        const float scale = pad_info[2];
        CHECK(cudaEventRecord(end, stream));
        CHECK(cudaEventSynchronize(end));
        CHECK(cudaEventElapsedTime(&ms, start, end));
        std::cout << mBufferName << "costs time: " << ms << std::endl;
        
        at::Tensor proto, preds;
        // yolov5s-seg fp16
        if (m_is_seg)
        {
            // "output0" 1x25200x117
            // "output1" 1x32x160x160
            auto dim0 = mEngine->getTensorShape("output0");
            auto dim1 = mEngine->getTensorShape("output1");
            //dims2str(dim0);
            //dims2str(dim1);
            proto = at::zeros({ dim1.d[0], dim1.d[1], dim1.d[2], dim1.d[3] }, torch::kHalf);
            preds = at::zeros({ dim0.d[0], dim0.d[1], dim0.d[2] }, torch::kHalf);
            // at::Half
            mInOut["output1"]->device2host((void*)(proto.data_ptr<at::Half>()), stream);//
            mInOut["output0"]->device2host((void*)(preds.data_ptr<at::Half>()), stream);//
            
            // Wait for the work in the stream to complete
            CHECK(cudaStreamSynchronize(stream));
            start_t = steady_clock::now();
            //proto = proto.to(torch::kCUDA);
            //preds = preds.to(torch::kCUDA);
            auto detections = algorithms::non_max_suppression(preds, 0.25, 0.45, 32);
            
            auto bs = detections.size(0);   // batch size
            for (int i = 0; i < bs; i++)
            {
                auto det = detections[i];
                // mask
                auto masks = process_mask(proto[i], detections[i].slice(1, 6), detections[i].slice(1, 0, 4), { img.rows, img.cols }, true);
                auto results = plot_masks(masks, { frame.rows, frame.cols }, pad_w, pad_h, scale, inputs.squeeze(0), 0.5);
                auto t_img = results.clamp(0, 255).to(torch::kU8);
                cv::Mat img_(t_img.size(0), t_img.size(1), CV_8UC3, t_img.data_ptr<uchar>());

                algorithms::scale_boxes(det, pad_w, pad_h, scale, cv::Size(img_.cols, img_.rows));
                bool draw = true;
                if (draw)
                {
                    for (int i = 0; i < det.size(0); i++)
                    {
                        auto x1 = det[i][0].item().toFloat();
                        auto y1 = det[i][1].item().toFloat();
                        auto x2 = det[i][2].item().toFloat();
                        auto y2 = det[i][3].item().toFloat();
                        auto score = det[i][4].item().toFloat();
                        auto cls = det[i][5].item().toInt();
                        cv::Point p1(x1, y1);
                        cv::Point p2(x2, y2);
                        cv::rectangle(img_, p1, p2,
                            cv::Scalar(255, 0, 0));
                        char s_buffer[20];  // maximum expected length of the float
                        std::snprintf(s_buffer, 20, "%.1f", score);
                        std::string s_str(s_buffer);
                        std::string label = s_str + " " + names[cls];
                        int baseline;
                        cv::Size textSize = cv::getTextSize(label, 0, 1, 2, &baseline);
                        p2 = cv::Point(p1.x + textSize.width, p1.y - textSize.height);
                        cv::rectangle(img_, p1, p2,
                            cv::Scalar(255, 0, 0));
                        cv::putText(img_, label, p1, 0, 1, cv::Scalar(255, 0, 0));
                    }
                    cv::imwrite(mBufferName + "_output.jpg", img_);
                }
            }
        }
        // yolov5s fp16
        else
        {
            auto dim0 = mEngine->getTensorShape("output0");
            preds = at::zeros({ dim0.d[0], dim0.d[1], dim0.d[2] }, torch::kHalf);
            mInOut["output0"]->device2host((void*)(preds.data_ptr<at::Half>()), stream);

            // Wait for the work in the stream to complete
            CHECK(cudaStreamSynchronize(stream));
            start_t = steady_clock::now();
            auto detections = algorithms::non_max_suppression(preds, 0.25, 0.45, 0);
            auto bs = detections.size(0);   // batch size
            for (int i = 0; i < bs; i++)
            {
                auto det = detections[i];
                algorithms::scale_boxes(det, pad_w, pad_h, scale, cv::Size(frame.cols, frame.rows));
                for (int i = 0; i < det.size(0); i++)
                {
                    auto x1 = det[i][0].item().toFloat();
                    auto y1 = det[i][1].item().toFloat();
                    auto x2 = det[i][2].item().toFloat();
                    auto y2 = det[i][3].item().toFloat();
                    auto score = det[i][4].item().toFloat();
                    auto cls = det[i][5].item().toInt();
                    // std::cout << "score: " << score << "cls " << cls << std::endl;
                    cv::Point p1(x1, y1);
                    cv::Point p2(x2, y2);
                    cv::rectangle(frame, p1, p2,
                        cv::Scalar(255, 0, 0));
                    char s_buffer[20];  // maximum expected length of the float
                    std::snprintf(s_buffer, 20, "%.1f", score);
                    std::string s_str(s_buffer);
                    std::string label = s_str + " " + names[cls];
                    int baseline;
                    cv::Size textSize = cv::getTextSize(label, 0, 1, 2, &baseline);
                    p2 = cv::Point(p1.x + textSize.width, p1.y - textSize.height);
                    cv::rectangle(frame, p1, p2,
                        cv::Scalar(255, 0, 0));
                    cv::putText(frame, label, p1, 0, 1, cv::Scalar(255, 0, 0));
                }
                cv::imwrite(mBufferName + "_output.jpg", frame);
            }
        }
        
        steady_clock::time_point end_t = steady_clock::now();
        long long diff = (std::chrono::time_point_cast<milliseconds>(end_t) -
            std::chrono::time_point_cast<milliseconds>(start_t))
            .count();
        std::cout << mBufferName << " verifyOutput costs time: " << diff << std::endl;
        
        return 0;
    }
public:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;
    cudaEvent_t start, end;
    
    std::vector<void*> mDeviceBindings;
    std::map<std::string, std::unique_ptr<algorithms::DeviceBuffer>> mInOut;
    std::vector<float> pad_info;
    torch::Tensor inputs;
    std::vector<std::string> names;
    cv::Mat frame;
    cv::Mat img;
    int inp_width = 640;
    int inp_height = 640;
    std::string mBufferName;
    bool m_is_seg;
};



// namespace std
// {
// ostream &operator<<(ostream &out, const pair<string, string> &value)
// {
//     return out << value.first << " " << value.second;
// }
// } // namespace std

auto prepareInput = [](std::shared_ptr<InferBuffer> ibuffer)
{
    ibuffer->prepareInput();
    return ibuffer;
};

auto infer = [](std::shared_ptr<InferBuffer> ibuffer)
{
    ibuffer->infer();
    return ibuffer;
};

auto verifyOutput = [](std::shared_ptr<InferBuffer> ibuffer) {
    ibuffer->verifyOutput();
    return ibuffer;
};

void onnx2engine(std::string input_f, std::string output_f, bool is_seg=true)
{
    const std::string modelFile(input_f);
    std::unique_ptr<nvinfer1::IBuilder> builder(createInferBuilder(logger));
    // create a network definition
    // The kEXPLICIT_BATCH flag is required in order to import models using the ONNX parser.
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    //auto network = std::make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flag));

    // Importing a Model Using the ONNX Parser
    //auto parser = std::make_unique<nvonnxparser::IParser>(createParser(*network, logger));
    std::unique_ptr<nvonnxparser::IParser> parser(createParser(*network, logger));

    // read the model file and process any errors
    parser->parseFromFile(modelFile.c_str(),
                          static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // create a build configuration specifying how TensorRT should optimize the model
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

    // maximum workspace size
    int workspace = 4;  // GB
    config->setMaxWorkspaceSize(workspace * 1U << 30);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    //config->setAvgTimingIterations(1);
    //config->setMinTimingIterations(1);

    config->setFlag(BuilderFlag::kFP16);
    //builder->setMaxBatchSize(4);
    // YoloInt8EntropyCalibrator2 calibrator("1","images");
    // config->setInt8Calibrator(&calibrator);
    if(is_seg)
    {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kMIN, {4, 1,3,640,640 });
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kOPT, {4, 1,3,640,640 });
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kMAX, {4,1,3,640,640 });

        profile->setDimensions("output1", nvinfer1::OptProfileSelector::kMIN, { 4, 1,32,160,160 });
        profile->setDimensions("output1", nvinfer1::OptProfileSelector::kOPT, { 4, 1,32,160,160 });
        profile->setDimensions("output1", nvinfer1::OptProfileSelector::kMAX, { 4, 1,32,160,160 });

        profile->setDimensions("output0", nvinfer1::OptProfileSelector::kMIN, { 3, 1,25200,117});
        profile->setDimensions("output0", nvinfer1::OptProfileSelector::kOPT, { 3, 1,25200,117 });
        profile->setDimensions("output0", nvinfer1::OptProfileSelector::kMAX, { 3, 1,25200,117 });

        config->addOptimizationProfile(profile);
    }
    else
    {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kMIN, {4, 1,3,640,640 });
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kOPT, {4, 1,3,640,640 });
        profile->setDimensions("images", nvinfer1::OptProfileSelector::kMAX, {4,1,3,640,640 });

        profile->setDimensions("output0", nvinfer1::OptProfileSelector::kMIN, { 3, 1,25200,85});
        profile->setDimensions("output0", nvinfer1::OptProfileSelector::kOPT, { 3, 1,25200,85 });
        profile->setDimensions("output0", nvinfer1::OptProfileSelector::kMAX, { 3, 1,25200,85 });

        config->addOptimizationProfile(profile);
    }
    

    // create an engine
    // auto serializedModel = std::make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
     std::unique_ptr<nvinfer1::IHostMemory> serializedModel(builder->buildSerializedNetwork(*network, *config));
     std::cout << "serializedModel->size()" << serializedModel->size() << std::endl;
     std::ofstream outfile(output_f, std::ofstream::out | std::ofstream::binary);
     outfile.write((char*)serializedModel->data(), serializedModel->size());
}

int main(int argc, char const *argv[])
{
    /* code */
    const cv::String keys =
    "{help h usage ? | | sampleAsyscTRTYolo -is_seg=1|0 }"
    "{is_seg |true | segment or not }"
    ;
    cv::CommandLineParser parser(argc, argv, keys);
    bool is_seg = parser.get<bool>("is_seg");
    std::cout << "-is_seg " << is_seg << std::endl;
    std::string input_f,output_f;
    // 
    // 
    if(is_seg)
    {
        input_f = "/workspace/runtimeDL/data/yolov5s-seg.onnx";
        output_f = "yolov5s-seg.engine";
    }
    else
    {
        input_f = "/workspace/runtimeDL/data/yolov5s.onnx";
        output_f = "yolov5s.engine";
    }
    onnx2engine(input_f,output_f,is_seg);

    ThreadPool pool(4);

    const std::string modelFile = output_f;
    std::ifstream engineFile(modelFile.c_str(), std::ifstream::binary);
    assert(engineFile);

    int fsize;
    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

     if (engineFile)
      std::cout << "all characters read successfully." << std::endl;
    else
      std::cout << "error: only " << engineFile.gcount() << " could be read" << std::endl;
    engineFile.close();

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));

    cv::Mat frame = cv::imread("/workspace/runtimeDL/data/dog.jpg");

    std::vector< std::future<std::shared_ptr<InferBuffer> > > prepare_results, infer_results, verify_results;
    
    steady_clock::time_point start_t = steady_clock::now();
    // prepareInput
    for (int i = 0; i < 4; ++i) {
        std::shared_ptr<InferBuffer> b(new InferBuffer(std::to_string(i), mEngine, frame,is_seg));
        prepare_results.emplace_back(
            pool.enqueue(prepareInput, b)
        );
    }

    // infer
    for (auto&& result : prepare_results)
        infer_results.emplace_back(
            pool.enqueue(infer, result.get())
        );

    // verifyOutput
    for (auto&& result : infer_results)
        verify_results.emplace_back(
            pool.enqueue(verifyOutput, result.get())
        );

    // result
    for (auto&& result : verify_results)
        result.get();

    steady_clock::time_point end_t = steady_clock::now();
    long long diff = (std::chrono::time_point_cast<milliseconds>(end_t) -
        std::chrono::time_point_cast<milliseconds>(start_t))
        .count();
    std::cout <<" total costs time: " << diff << std::endl;

    return 0;
}
