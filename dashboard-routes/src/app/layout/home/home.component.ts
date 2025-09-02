import { Component, OnInit, ViewChild } from '@angular/core';
import { DeliveryPlanService, VRPSolution, DetailedRoute } from '../../services/delivery-plan.service';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTableModule } from '@angular/material/table';
import { MatSort, MatSortModule } from '@angular/material/sort';
import { MatButtonModule } from '@angular/material/button';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatIconModule,
    MatListModule,
    MatProgressBarModule,
    MatTableModule,
    MatSortModule,
    MatButtonModule
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent implements OnInit {
  data?: VRPSolution;
  isLoading = true;
  errorMessage: string | null = null;

  totalDistance = 0;
  vehiclesUsed = 0;

  displayedColumns: string[] = ['vehicle_id', 'vehicle_type', 'path', 'total_demand', 'distance_km', 'route_cost'];
  dataSource: DetailedRoute[] = [];

  @ViewChild(MatSort) sort!: MatSort;

  constructor(
    private deliveryPlanService: DeliveryPlanService,
    private sanitizer: DomSanitizer
  ) {}

  ngOnInit(): void {
    this.loadSolution();
  }

  loadSolution(): void {
    this.isLoading = true;
    this.deliveryPlanService.getSolution().subscribe({
      next: (res: VRPSolution) => {
        this.data = res;
        this.dataSource = res.route_details;
        this.processDataForDashboard(res);
        this.isLoading = false;
        this.errorMessage = null;
      },
      error: (err) => {
        console.error('Error loading solution:', err);
        this.isLoading = false;
        this.errorMessage = 'Failed to load delivery plan. Please try again.';
      }
    });
  }

  generateNewSolution(): void {
    this.isLoading = true;
    this.deliveryPlanService.getGenerateNewCustomers().subscribe({
      next: () => {
        this.loadSolution();
      },
      error: (err) => {
        console.error('Error generating new customers:', err);
        this.isLoading = false;
        this.errorMessage = 'Failed to generate new data. Please try again.';
      }
    });
  }

  getSanitizedImageUrl(base64String: string): SafeResourceUrl {
    return this.sanitizer.bypassSecurityTrustResourceUrl('data:image/png;base64,' + base64String);
  }
  
  private processDataForDashboard(data: VRPSolution): void {
    this.totalDistance = data.route_details.reduce((sum, route) => sum + route.distance_km, 0);
    this.vehiclesUsed = data.route_details.length;
  }
}