#include "headposecomponent.h"
#include "infer.h"


TRTEngineSingleton* TRTEngineSingleton::instance = nullptr;

//constructor
HeadPoseComponent::HeadPoseComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<std::string>& outputQueue,ThreadSafeQueue<cv::Mat>& framesQueue, ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), outputQueue(outputQueue),framesQueue(framesQueue) ,commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}


//destructor
HeadPoseComponent::~HeadPoseComponent() {
    stopHeadPoseDetection();
}

// initialize model 
bool HeadPoseComponent::initialize() {
    return true;
}

// start detection loop in another thread
void HeadPoseComponent::startHeadPoseDetection() {
    if (running) {
        std::cerr << "headpose Detection is already running." << std::endl;
        return;
    }
    running = true;
    HeadPoseDetectionThread = std::thread(&HeadPoseComponent::HeadPoseDetectionLoop, this);
}

// release thread and any needed cleanup
void HeadPoseComponent::stopHeadPoseDetection() {
    running = false;
    if (HeadPoseDetectionThread.joinable()) {
        HeadPoseDetectionThread.join();
    }
}


// This loop takes frame from input queue , sends it to detect faces and places it into the output queue
void HeadPoseComponent::HeadPoseDetectionLoop() {
    cv::Mat frame;
    this->lastTime = std::chrono::high_resolution_clock::now(); // Initialize the last time

    while (running) {
        if (inputQueue.tryPop(frame)) {



            auto start = std::chrono::high_resolution_clock::now();
            detectHeadPose(frame);
            auto end = std::chrono::high_resolution_clock::now();

            double detectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            updatePerformanceMetrics(detectionTime);
            displayPerformanceMetrics(frame);





            framesQueue.push(frame);
        }
    }
}






//function to start the head pose detection
void HeadPoseComponent::detectHeadPose(cv::Mat& frame) {

    TRTEngineSingleton* trt=TRTEngineSingleton::getInstance();
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> out = trt->infer(frame);
    auto end = std::chrono::high_resolution_clock::now();
    double engineTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "head pose time:" << std::endl;
    std::cout << engineTime << std::endl;
    //for (float x : out){
	//printf("%f \n",x);
	//}
	
   //outputQueue.push(out);

}





void HeadPoseComponent::updatePerformanceMetrics(double detectionTime) {
    totalDetectionTime += detectionTime;  // Total time spent on detection
    totalFramesProcessed++;               // Increment the frame count

    // Calculation of average time per frame and updating fps immediately
    avgDetectionTime = totalDetectionTime / totalFramesProcessed;
    auto currentTime = std::chrono::high_resolution_clock::now();
    fps = totalFramesProcessed / (std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastTime).count() + 1e-9);  // Adding a small epsilon to avoid division by zero
}

void HeadPoseComponent::displayPerformanceMetrics(cv::Mat& frame) {
    std::string fpsText = "FPS: " + std::to_string(int(fps));
    std::string avgTimeText = "Avg Time per Frame: " + std::to_string(avgDetectionTime) + " ms";

    // Output the metrics to the console; you might want to output to the image or a GUI in a real application
    //std::cout << fpsText << std::endl;
    //std::cout << avgTimeText << std::endl;
}


