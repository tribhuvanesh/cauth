#!/bin/bash
g++ -g -c soft.cpp  -I/usr/include/opencv -lml -lcvaux -lhighgui -lcv -lcxcore 
echo "Compiled soft.cpp"
g++ -g -c utils.cpp  -I/usr/include/opencv -lml -lcvaux -lhighgui -lcv -lcxcore 
echo "Compiled utils.cpp"
g++ -g -c detect.cpp  -I/usr/include/opencv -lml -lcvaux -lhighgui -lcv -lcxcore 
echo "Compiled detect.cpp"
g++ -g -c main.cpp  -I/usr/include/opencv -lml -lcvaux -lhighgui -lcv -lcxcore 
echo "Compiled main.cpp"
g++ -g -o main main.o detect.o utils.o soft.o libsvm-3.12/svm.cpp -lcryptopp -lpthread `pkg-config --cflags --libs opencv`
echo "Completed build. Run ./main to execute the program"
