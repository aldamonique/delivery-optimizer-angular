import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';


export interface DetailedRoute {
  vehicle_id: number;
  vehicle_type: string;
  path: number[];
  total_demand: number;
  distance_km: number;
  route_cost: number;
  arrival_times: string[];
}
export interface TimeWindow {
  start: number;
  end: number;
}
export interface PointData {
  id: number;
  x: number;
  y: number;
  demand: number;
  window: TimeWindow;
}
export interface CustomerData{
  depot: PointData;
  customers: PointData[];
}

export interface GenerationStats {
  generation_number: number;
  best_cost: number;
  average_cost: number;
}

export interface VRPSolution {
  best_total_cost: number;
  found_at_generation: number;
  route_details: DetailedRoute[];
  generation_history: GenerationStats[];
  solution_image_base64: string;
  timestamp?: string;
}


@Injectable({
  providedIn: 'root'
})
export class DeliveryPlanService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  getSolution():Observable<any>{
    return this.http.get<VRPSolution>(`${this.apiUrl}/solve`);
  }
  getCustomers():Observable<any>{
    return this.http.get<CustomerData>(`${this.apiUrl}/customer`);
  }

  
  getGenerateNewCustomers():Observable<any>{
    return this.http.get<CustomerData>(`${this.apiUrl}/customer/generate`);
  }
}
