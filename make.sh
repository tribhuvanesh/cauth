#!/bin/bash
g++ -c soft.cpp -lcryptopp -lpthread `pkg-config --cflags --libs opencv`
g++ -c utils.cpp -lcryptopp -lpthread `pkg-config --cflags --libs opencv`
g++ -c detect.cpp -lcryptopp -lpthread `pkg-config --cflags --libs opencv`
g++ -c main.cpp -lcryptopp -lpthread `pkg-config --cflags --libs opencv`
g++ -o main main.o detect.o utils.o soft.o -lcryptopp -lpthread `pkg-config --cflags --libs opencv`
