# 🚚 Vehicle Routing Problem (VRP) Solver
A Vehicle Route Optimizer with a Genetic Algorithm, FastAPI, and Angular.
This project is a full-stack solution for the Vehicle Routing Problem (VRP), developed as the final project for the Artificial Intelligence course at the Instituto Federal da Bahia (IFBA).

The application combines a robust Python backend with FastAPI, which uses a Genetic Algorithm to find optimal solutions, and an interactive Angular frontend that allows for visualizing and interacting with the results. The system is designed to solve complex combinatorial optimization problems by considering real-world constraints such as vehicle capacities, operational costs, and customer time windows.

## 📸 Screenshot

A quick glance at the application's dashboard, showcasing an optimized delivery plan and its key metrics.


<img width="1882" height="1072" alt="image" src="https://github.com/user-attachments/assets/89ada825-d330-4560-b47f-c094bc5e3ee1" />
 
## ✨ Features

- Optimization Backend: An API built with FastAPI that implements a Genetic Algorithm to solve the VRP with Time Windows (VRPTW).

- Interactive Dashboard: A responsive UI developed in Angular to visualize optimized routes, cost metrics, and efficiency.

- Dynamic Scenario Generation: Functionality to generate new problem instances with random customers and demands directly from the UI.

- Graphical Visualization: Generation of optimized route plots using Matplotlib, clearly displayed on the dashboard.

- Detailed Metrics: Display of essential KPIs, such as total cost, total distance, vehicles used, and details for each individual route.

## 🛠️ Technologies Used

| Category    | Technology                                                                                                  |
| :---------- | :---------------------------------------------------------------------------------------------------------- |
| **Backend** | **Python 3**, **FastAPI** (for the API), **Uvicorn** (ASGI server), **Matplotlib** (for plot generation)      |
| **Frontend** | **Angular**, **TypeScript**, **Angular Material** (for UI components), **RxJS** (for reactive programming)    |




### 🚧 Project Status
This project is currently in active development. Next steps include implementing unit tests, algorithm optimizations, and adding new features to the dashboard.
