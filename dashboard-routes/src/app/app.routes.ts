import { Routes } from '@angular/router';
import { HomeComponent } from './layout/home/home.component';
import { NotFoundComponent } from './error-pages/not-found/not-found.component';
export const routes: Routes = [
    {
        path: '',
        component: HomeComponent, 
        pathMatch:'full'
        
    }, 
    {
        path:'home', 
        component: HomeComponent
    }, 
    {
        path: '**', 
        component: NotFoundComponent
    }
];
