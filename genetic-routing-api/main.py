import random
import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# --- API Metadata ---
app = FastAPI(
    title="Vehicle Routing Problem (VRP) Optimization API",
    description="Uses a Genetic Algorithm with Time Windows to solve the VRP.",
    version="3.0.0"
)

origins = ["http://localhost:4200"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class TimeWindow(BaseModel):
    start: float
    end: float

class PointData(BaseModel):
    id: int
    x: int
    y: int
    demand: int
    window: TimeWindow

class ProblemData(BaseModel):
    depot: PointData
    customers: List[PointData]

class DetailedRoute(BaseModel):
    vehicle_id: int
    vehicle_type: str
    path: List[int]
    total_demand: int
    distance_km: float
    route_cost: float
    arrival_times: List[str]

class GenerationStats(BaseModel):
    generation_number: int
    best_cost: float
    average_cost: float

class VRPSolution(BaseModel):
    best_total_cost: float
    found_at_generation: int
    route_details: List[DetailedRoute]
    generation_history: List[GenerationStats]

class ErrorMessage(BaseModel):
    detail: str

# --- Data and Constants ---
VEHICLES = [
    {'id': 1, 'type': 'small', 'capacity': 60, 'cost_km': 1.2, 'speed_kmh': 40},
    {'id': 2, 'type': 'medium', 'capacity': 80, 'cost_km': 1.5, 'speed_kmh': 40},
    {'id': 3, 'type': 'large', 'capacity': 100, 'cost_km': 2.0, 'speed_kmh': 40}
]
DEPOT = {'id': 0, 'x': 50, 'y': 50, 'janela': (0, 24)} # Depósito aberto 24h
SERVICE_TIME_HOURS = 0.25  # 15 minutos de tempo de serviço por cliente
TIME_PENALTY_FACTOR = 100 # Penalidade alta por violar a janela de tempo

# --- Genetic Algorithm Parameters ---
NUM_CUSTOMERS = 20
MAP_X = 100
MAP_Y = 100
POPULATION_SIZE = 150
NUM_GENERATIONS = 1000
MUTATION_RATE = 0.5
NUM_ELITE = 15

def generate_customers(num_customers, map_x, map_y):
    customers = []
    for i in range(1, num_customers + 1):
        start_window = random.randint(8, 14)  # Janela começa entre 8h e 14h
        end_window = start_window + random.randint(2, 4)
        customers.append({
            'id': i, 'x': random.randint(5, map_x - 5), 'y': random.randint(5, map_y - 5),
            'demand': random.randint(5, 15),
            'window': (start_window, end_window)
        })
    return customers

CUSTOMER_LIST = generate_customers(NUM_CUSTOMERS, MAP_X, MAP_Y)
DATA_MAP = {customer['id']: customer for customer in CUSTOMER_LIST}
DATA_MAP[0] = DEPOT

# --- Genetic Algorithm Functions ---
def calculate_distance(id1, id2):
    point1 = DATA_MAP[id1]
    point2 = DATA_MAP[id2]
    return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

def calculate_route_cost_and_penalty(route, vehicle):
    if not route:
        return 0, []

    cost = 0
    time_penalty = 0
    current_time = 0
    last_location_id = DEPOT['id']
    arrival_times = []

    for customer_id in route:
        distance = calculate_distance(last_location_id, customer_id)
        travel_time = distance / vehicle['speed_kmh']
        current_time += travel_time
        
        customer_window = DATA_MAP[customer_id]['window']
        
        # Espera se chegar cedo
        if current_time < customer_window[0]:
            current_time = customer_window[0]
        
        # Penaliza se chegar atrasado
        if current_time > customer_window[1]:
            time_penalty += (current_time - customer_window[1]) * TIME_PENALTY_FACTOR

        arrival_times.append(f"{int(current_time)}:{int((current_time*60)%60):02d}")
        current_time += SERVICE_TIME_HOURS
        last_location_id = customer_id

    # Custo de volta ao depósito
    distance_to_depot = calculate_distance(last_location_id, DEPOT['id'])
    cost = (calculate_distance(DEPOT['id'], route[0]) + sum(calculate_distance(route[i], route[i+1]) for i in range(len(route)-1)) + distance_to_depot) * vehicle['cost_km']
    
    total_cost = cost + time_penalty
    return total_cost, arrival_times


def create_individual():
    customer_ids = [customer['id'] for customer in CUSTOMER_LIST]
    random.shuffle(customer_ids)
    return customer_ids

def create_initial_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def calculate_fitness(individual):
    shuffled_vehicles = VEHICLES[:]
    random.shuffle(shuffled_vehicles)
    routes_by_vehicle = {}
    allocated_customers = set()

    for vehicle in shuffled_vehicles:
        current_route = []
        current_demand = 0
        routes_by_vehicle[vehicle['id']] = current_route
        for customer_id in individual:
            if customer_id not in allocated_customers:
                customer_demand = DATA_MAP[customer_id]['demand']
                if current_demand + customer_demand <= vehicle['capacity']:
                    current_route.append(customer_id)
                    current_demand += customer_demand
                    allocated_customers.add(customer_id)

    if len(allocated_customers) < NUM_CUSTOMERS:
        return 0, {}

    total_cost = 0
    for v_id, route in routes_by_vehicle.items():
        vehicle = next(v for v in VEHICLES if v['id'] == v_id)
        route_cost, _ = calculate_route_cost_and_penalty(route, vehicle)
        total_cost += route_cost

    return 1 / (1 + total_cost), routes_by_vehicle

def tournament_selection(evaluated_population):
    participants = random.sample(evaluated_population, 5)
    participants.sort(key=lambda item: item[1], reverse=True)
    return participants[0][0]

def crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end+1] = parent1[start:end+1]
    
    parent2_genes = [gene for gene in parent2 if gene not in child]
    
    child_idx = 0
    for gene in parent2_genes:
        while child[child_idx] is not None:
            child_idx += 1
        child[child_idx] = gene
    return child

def apply_mutation(individual):
    if random.random() < MUTATION_RATE:
        mutation_type = random.random()
        if mutation_type < 0.4:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual[start:end+1] = reversed(individual[start:end+1])
        elif mutation_type < 0.8:
            customer_idx = random.randrange(len(individual))
            customer = individual.pop(customer_idx)
            insert_point = random.randrange(len(individual) + 1)
            individual.insert(insert_point, customer)
        else:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

# --- Main Solver Function ---
def solve_vrp():
    population = create_initial_population()
    global_best_solution = None
    global_lowest_cost = float('inf')
    best_cost_generation = -1
    generation_history = []

    for i in range(NUM_GENERATIONS):
        evaluated_population = []
        total_cost_sum = 0
        valid_solutions_count = 0

        for individual in population:
            fitness, routes = calculate_fitness(individual)
            cost = (1 / fitness) - 1 if fitness > 0 else float('inf')
            evaluated_population.append((individual, fitness, cost, routes))
            if cost != float('inf'):
                total_cost_sum += cost
                valid_solutions_count += 1

        evaluated_population.sort(key=lambda item: item[1], reverse=True)
        
        current_best_cost_in_gen = evaluated_population[0][2]
        if current_best_cost_in_gen < global_lowest_cost:
            global_lowest_cost = current_best_cost_in_gen
            global_best_solution = evaluated_population[0][3]
            best_cost_generation = i

        average_cost = (total_cost_sum / valid_solutions_count) if valid_solutions_count > 0 else 0
        
        if i == 1 or (i+1) %20 ==0 or i == NUM_GENERATIONS :
            generation_history.append(
                GenerationStats(
                    generation_number=i,
                    best_cost=round(global_lowest_cost, 2),
                    average_cost=round(average_cost, 2)
                )
            )

        elite = [ind[0] for ind in evaluated_population[:NUM_ELITE]]
        new_population = elite
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(evaluated_population)
            parent2 = tournament_selection(evaluated_population)
            child = crossover(parent1, parent2)
            child = apply_mutation(child)
            new_population.append(child)
        
        population = new_population

    if global_best_solution is None:
        return None

    return {
        "best_cost_generation": best_cost_generation,
        "global_best_solution": global_best_solution,
        "generation_history": generation_history
    }

# --- API Endpoints ---
@app.get("/", summary="Welcome Message", include_in_schema=False)
def root():
    return {"message": "Welcome to the VRP with Time Windows API"}


@app.get("/customer", response_model=ProblemData,
         summary="Get current problem data (depot and costumers)")

def get_customers():

    global CUSTOMER_LIST, DATA_MAP
    DATA_MAP = {customer['id']: customer for customer in CUSTOMER_LIST}
    DATA_MAP[0] = DEPOT 

    depot_data = PointData(
        id=DEPOT['id'],
        x=DEPOT['x'],
        y=DEPOT['y'],
        demand=0,
        window=TimeWindow(start=DEPOT['janela'][0], end=DEPOT['janela'][1])
    )

    customer_data_list = [
        PointData(
            id=c['id'],
            x=c['x'],
            y=c['y'],
            demand=c['demand'],
            window=TimeWindow(start=c['window'][0], end=c['window'][1])
        ) for c in CUSTOMER_LIST
    ]

    return ProblemData(depot=depot_data, customers=customer_data_list)



@app.get("/customer/generate", response_model=ProblemData,
         summary="Get current problem data (depot and costumers)")

def get_customers():

    global CUSTOMER_LIST, DATA_MAP

    CUSTOMER_LIST = generate_customers(NUM_CUSTOMERS, MAP_X, MAP_Y)

    DATA_MAP = {customer['id']: customer for customer in CUSTOMER_LIST}
    DATA_MAP[0] = DEPOT 

    depot_data = PointData(
        id=DEPOT['id'],
        x=DEPOT['x'],
        y=DEPOT['y'],
        demand=0,
        window=TimeWindow(start=DEPOT['janela'][0], end=DEPOT['janela'][1])
    )

    customer_data_list = [
        PointData(
            id=c['id'],
            x=c['x'],
            y=c['y'],
            demand=c['demand'],
            window=TimeWindow(start=c['window'][0], end=c['window'][1])
        ) for c in CUSTOMER_LIST
    ]

    return ProblemData(depot=depot_data, customers=customer_data_list)



@app.get("/solve",
         response_model=VRPSolution,
         summary="Executes the GA to find the best solution with time windows",
         responses={404: {"model": ErrorMessage, "description": "Solution not found"}})
def solve():
    execution_result = solve_vrp()

    if not execution_result:
        raise HTTPException(status_code=404, detail="Could not find a viable solution.")

    final_total_cost = 0
    detailed_routes = []
    
    solution = execution_result["global_best_solution"]
    
    for vehicle_id, route in solution.items():
        if not route:
            continue
            
        used_vehicle = next(v for v in VEHICLES if v['id'] == vehicle_id)
        demand = sum(DATA_MAP[cid]['demand'] for cid in route)
        cost, arrival_times = calculate_route_cost_and_penalty(route, used_vehicle)
        
        distance = 0
        if route:
            distance += calculate_distance(DEPOT['id'], route[0])
            for i in range(len(route) - 1):
                distance += calculate_distance(route[i], route[i+1])
            distance += calculate_distance(route[-1], DEPOT['id'])
        
        final_total_cost += cost

        detailed_routes.append(
            DetailedRoute(
                vehicle_id=vehicle_id,
                vehicle_type=used_vehicle['type'],
                path=route,
                total_demand=demand,
                distance_km=round(distance, 2),
                route_cost=round(cost, 2),
                arrival_times=arrival_times
            )
        )

    return VRPSolution(
        best_total_cost=round(final_total_cost, 2),
        found_at_generation=execution_result["best_cost_generation"],
        route_details=detailed_routes,
        generation_history=execution_result["generation_history"]
    )