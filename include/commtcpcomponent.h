#pragma once

#include <thread>
#include <atomic>
#include "threadsafequeue.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

class CommTCPComponent {
public:
    CommTCPComponent(int port, ThreadSafeQueue<cv::Mat>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue);
    ~CommTCPComponent();

    void startServer();
    void stopServer();
    void sendFrame(const cv::Mat& frame);

private:
    int port;
    std::atomic<bool> running;
    std::thread serverThread;
    ThreadSafeQueue<cv::Mat>& outputQueue;  // Queue for sending frames to connected client
    ThreadSafeQueue<std::string>& commandsQueue;  // Queue for sending commands to dms manager
    ThreadSafeQueue<std::string>& faultsQueue;

    void serverLoop();  // Main loop for accepting connections and processing commands
    void handleClient(int clientSocket);  // Handle communication with a connected client
};

