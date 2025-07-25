import { Component, HostBinding, Renderer2, OnInit, PLATFORM_ID, Inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';


//componente flutuante
import {OverlayContainer} from '@angular/cdk/overlay'

import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatIconModule} from '@angular/material/icon';
import {MatToolbarModule} from '@angular/material/toolbar';
import { FormsModule } from '@angular/forms';
import { isPlatformBrowser, JsonPipe } from '@angular/common';
import { NavbarComponent } from "./layout/navbar/navbar.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, NavbarComponent, MatIconModule, MatSlideToggleModule, MatToolbarModule, FormsModule, NavbarComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'Fast Delivery';
  isDarkTheme = false;

  constructor(
    @Inject(PLATFORM_ID) private platformId: Object,
    private renderer: Renderer2, 
    private overlayContainer: OverlayContainer){

  }
  //altera a classe do proprio host do componente
 /// @HostBinding('class.dark-theme') get isDark():boolean{
  //  return this.isDarkTheme;
  //}
    ngOnInit(): void {
    if (isPlatformBrowser(this.platformId)) {
      const savedTheme = localStorage.getItem('isDarkTheme');
      if (savedTheme) {
        this.isDarkTheme = JSON.parse(savedTheme);
       // this.updateTheme(); 
      }
    }
  }
  
/*
  toggleTheme():void{
    if(isPlatformBrowser(this.platformId)){
      this.isDarkTheme= !this.isDarkTheme;
      this.updateTheme();
      localStorage.setItem('isDarkTheme', JSON.stringify(this.isDarkTheme));
    }
  }

private updateTheme(): void {
  const themeClass = 'dark-theme';
  const htmlElement = document.documentElement; // <html>
  const overlayContainerElement = this.overlayContainer.getContainerElement();

  if (this.isDarkTheme) {
    this.renderer.addClass(htmlElement, themeClass);
    overlayContainerElement.classList.add(themeClass);
    console.log('Dark theme applied');
  } else {
    this.renderer.removeClass(htmlElement, themeClass);
    overlayContainerElement.classList.remove(themeClass);
    console.log('Light theme applied');
  }
}
*/

}
