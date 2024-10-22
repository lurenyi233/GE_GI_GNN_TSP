	
README
	
This file contains seven folders each containing 190 Symmetric TSP instances with 100 cities.
The folders are labelled:
RANDOM: 190 TSP instances with 100 cities placed randomly in 2-d plane
CLKeasy: 190 TSP instances evolved to be easy for chained Lin-Kernighan (CLK) heuristic
CLKhard: 190 TSP instances evolved to be hard for chained Lin-Kernighan (CLK) heuristic
LKCCeasy: 190 TSP instances evolved to be easy for Lin-Kernighan with Cluster Compensation (LKCC) heuristic
LKCChard: 190 TSP instances evolved to be hard for Lin-Kernighan with Cluster Compensation (LKCC) heuristic
easyCLK-hardKLCC: 190 TSP instances evolved to be simultaneously easy for CLK and hard for LKCC
hardCLK-easyLKCC: 190 TSP instances evolved to be simultaneously hard for CLK and easy for LKCC

All instances have been evolved over 600 generations, with the objective of the evolutionary algorithm to maximise or minimise the mean search effort of 100 trials for
each heuristic.

The way these instances were created, and some of their properties, is
discussed in [1] and [2].

[1] J.I. van Hemert. Property analysis of symmetric travelling salesman problem
instances acquired through evolution. In G. Raidl and J. Gottlieb, editors,
Evolutionary Computation in Combinatorial Optimization, Springer Lecture Notes
on Computer Science, pages 122--131. Springer-Verlag, Berlin, 2005.

[2] Smith-Miles, K. & van Hemert, J. Discovering the suitability of optimisation algorithms by learning from evolved instances.
Ann Math Artif Intell (2011) 61: 87. doi:10.1007/s10472-011-9230-5

--

