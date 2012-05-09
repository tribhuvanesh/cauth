#!/bin/bash
g++ -c soft.cpp  -I/usr/include/opencv -lml -lcvaux -lhighgui -lcv -lcxcore 
echo "Compiled soft.cpp"
g++ -c utils.cpp  -I/usr/include/opencv -lml -lcvaux -lhighgui -lcv -lcxcore 
echo "Compiled utils.cpp"
g++ -c detect.cpp  -I/usr/include/opencv -lml -lcvaux -lhighgui -lcv -lcxcore 
echo "Compiled detect.cpp"
g++ -c main.cpp  -I/usr/include/opencv -lml -lcvaux -lhighgui -lcv -lcxcore 
echo "Compiled main.cpp"
g++ -o main main.o detect.o utils.o soft.o -lcryptopp -lpthread `pkg-config --cflags --libs opencv`
echo "Completed build. Run ./main to execute the program"
