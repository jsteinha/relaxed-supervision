SNOPT=~/snopt7
main: main.cpp Makefile
	g++ -O3 -I $(SNOPT)/include -c main.cpp -o main.o --std=c++11 -pthread
	g++ -O3 main.o  -o main -Wl,-rpath $(SNOPT)/lib -L $(SNOPT)/lib -lsnopt7_cpp -lsnopt7 --std=c++11 -pthread
