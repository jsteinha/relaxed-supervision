SNOPT=~/snopt7
main: main.cpp Makefile util
	g++ -O3 -I $(SNOPT)/include -c main.cpp -o main.o --std=c++11 -pthread
	g++ -O3 main.o  -o main util.o -Wl,-rpath $(SNOPT)/lib -L $(SNOPT)/lib -lsnopt7_cpp -lsnopt7 --std=c++11 -pthread
util: util.cpp Makefile
	g++ -O3 -c util.cpp -o util.o --std=c++11 -pthread
clean:
	rm *.o
