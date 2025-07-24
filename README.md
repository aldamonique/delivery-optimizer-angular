# Vehicle Routing Problem Solver with Genetic Algorithm  
*Final project for the Artificial Intelligence course at Instituto Federal da Bahia (IFBA)*

This project presents a Genetic Algorithm implementation designed to solve the Vehicle Routing Problem (VRP), a challenging combinatorial optimization problem commonly found in logistics and distribution.

The API is built with **FastAPI**, allowing easy integration with frontend applications (e.g., Angular) and providing route optimizations that consider vehicle capacities, costs, client demands, and delivery time windows.

---

## Project Context and Motivation

Developed as the final work for the Artificial Intelligence discipline in the Analysis and Systems Development course at IFBA, this project applies theoretical concepts of heuristic search and optimization algorithms to a real-world problem.

The goal is to generate efficient and cost-effective delivery routes that satisfy constraints such as vehicle capacity and client time windows, demonstrating practical knowledge of AI techniques.

---

## Features and Enhancements

- Implements a bio-inspired Genetic Algorithm to explore a large search space efficiently  
- Supports multiple vehicle types with different capacities and costs  
- Dynamically generates client locations, demands, and time windows  
- Evaluates route fitness based on total cost and feasibility  
- Provides detailed JSON responses for integration and Matplotlib-based route visualizations  
- Modular codebase allowing easy extension and customization  

---

## Technologies Used

- Python 3  
- FastAPI for API development  
- Uvicorn for asynchronous server  
- Matplotlib for graphical visualization  
- Standard Python libraries (`random`, `math`, etc.)  