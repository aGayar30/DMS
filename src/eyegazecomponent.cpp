#include "eyegazecomponent.h"
#include "inferE.h"


TRTEngineSingletonE* TRTEngineSingletonE::instance = nullptr;

//constructor
EyeGazeComponent::EyeGazeComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<std::string>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue)
: inputQueue(inputQueue), outputQueue(outputQueue),commandsQueue(commandsQueue),faultsQueue(faultsQueue), running(false) {}


//destructor
EyeGazeComponent::~EyeGazeComponent() {
    stopEyeGazeDetection();
}

// initialize model 
bool EyeGazeComponent::initialize() {
    return true;
}

// start detection loop in another thread
void EyeGazeComponent::startEyeGazeDetection() {
    if (running) {
        std::cerr << "eyegaze Detection is already running." << std::endl;
        return;
    }
    running = true;
    EyeGazeDetectionThread = std::thread(&EyeGazeComponent::EyeGazeDetectionLoop, this);
}

// release thread and any needed cleanup
void EyeGazeComponent::stopEyeGazeDetection() {
    running = false;
    if (EyeGazeDetectionThread.joinable()) {
        EyeGazeDetectionThread.join();
    }
}


// This loop takes frame from input queue , sends it to detect eyegaze and places it into the output queue
void EyeGazeComponent::EyeGazeDetectionLoop() {
    cv::Mat frame;
    this->lastTime = std::chrono::high_resolution_clock::now(); // Initialize the last time

    while (running) {
        if (inputQueue.tryPop(frame)) {

            std::cout << "about to start eye gaze" << std::endl;

            auto start = std::chrono::high_resolution_clock::now();
            detectEyeGaze(frame);
            auto end = std::chrono::high_resolution_clock::now();

            double detectionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            updatePerformanceMetrics(detectionTime);
            displayPerformanceMetrics(frame);





            //outputQueue.push(frame);
        }
    }
}






//function to start the head pose detection
void EyeGazeComponent::detectEyeGaze(cv::Mat& frame) {

    TRTEngineSingletonE* trt=TRTEngineSingletonE::getInstance();
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> out = trt->infer(frame);
    auto end = std::chrono::high_resolution_clock::now();
    double engineTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Eye gaze time:" << std::endl;
    std::cout << engineTime << std::endl;
    for (float x : out){
	printf("%f \n",x);
	}
	//
   //outputQueue.push(out);

}





void EyeGazeComponent::updatePerformanceMetrics(double detectionTime) {
    totalDetectionTime += detectionTime;  // Total time spent on detection
    totalFramesProcessed++;               // Increment the frame count

    // Calculation of average time per frame and updating fps immediately
    avgDetectionTime = totalDetectionTime / totalFramesProcessed;
    auto currentTime = std::chrono::high_resolution_clock::now();
    fps = totalFramesProcessed / (std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastTime).count() + 1e-9);  // Adding a small epsilon to avoid division by zero
}

void EyeGazeComponent::displayPerformanceMetrics(cv::Mat& frame) {
    std::string fpsText = "FPS: " + std::to_string(int(fps));
    std::string avgTimeText = "Avg Time per Frame: " + std::to_string(avgDetectionTime) + " ms";

    // Output the metrics to the console; you might want to output to the image or a GUI in a real application
    //std::cout << fpsText << std::endl;
    //std::cout << avgTimeText << std::endl;
}


