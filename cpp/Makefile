SNOPT=~/snopt7
STD=c++11
main: main.cpp Makefile util
	g++ -O3 -I $(SNOPT)/include -c main.cpp -o main.o --std=$(STD) -pthread
	g++ -O3 main.o  -o main util.o -Wl,-rpath $(SNOPT)/lib -L $(SNOPT)/lib -lsnopt7_cpp -lsnopt7 --std=$(STD) -pthread
util: util.cpp Makefile
	g++ -O3 -c util.cpp -o util.o --std=$(STD) -pthread
