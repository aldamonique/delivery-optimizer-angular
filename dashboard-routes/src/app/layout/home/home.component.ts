import { Component, OnInit} from '@angular/core';
import { CustomerData, DeliveryPlanService, PointData, VRPSolution } from '../../services/delivery-plan.service';
import { HttpClient } from '@angular/common/http';
import {MatProgressSpinner} from '@angular/material/progress-spinner'
import { MatCardModule } from '@angular/material/card';
import { MatDividerModule } from '@angular/material/divider';
import { MatListModule } from '@angular/material/list';
import { MatChipsModule } from '@angular/material/chips';
import { CommonModule } from '@angular/common'; 
import { MatTableModule } from '@angular/material/table';
import { MatIconModule } from '@angular/material/icon';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, MatProgressSpinner, MatIconModule, MatCardModule, MatTableModule, MatDividerModule, MatDividerModule, MatListModule, MatChipsModule],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})

export class HomeComponent implements OnInit{
  customerData?: PointData[];
  data?: VRPSolution;
  isLoading = true;
  errorMessage: string | null = null;


  displayedColumnsRoutes: String[] = ['vehicle_id', 'vehicle_type', 'path', 'total_demand', 'distance_km', 'route_cost'];
  displayedColumnsHistory: string[] = ['generation_number', 'best_cost', 'avarage_cost'];


  constructor(
    private deliveryPlanService:DeliveryPlanService, 
    private sanitizer: DomSanitizer){

  }

  ngOnInit(): void{
    this.loadSolution();
  
  }

  loadSolution(): void{
    this.deliveryPlanService.getSolution().subscribe({
      next: (res: VRPSolution) => {
        this.data = res;
        this.isLoading = false;
        this.errorMessage = null;

      },
      error: (err) => {
        console.log('Error loading solution: ', err);
        this.isLoading=false;
        this.errorMessage = 'Failed to load the delivery plan. Please try again.'
      }
    })
  }
  
  getGenerateNewCustomers(): void{
    this.isLoading = true;
    this.deliveryPlanService.getGenerateNewCustomers().subscribe({
      next: (res: PointData[]) => {
        this.customerData = res;
        this.loadSolution();
        this.getSanitizedImageUrl(this.data!.solution_image_base64);
      },
      error:(err) =>{
        console.error('Error for fetching data from FAST API: ', err);
        this.isLoading = false;
      }
    });
  }


  getSanitizedImageUrl(base64String: string): SafeResourceUrl {
    return this.sanitizer.bypassSecurityTrustResourceUrl('data:image/png;base64,' + base64String);
  }


}
