import { Component, ViewEncapsulation } from '@angular/core';
import {MatCardModule} from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { RouterLink } from '@angular/router';


@Component({
  selector: 'app-not-found',
  imports: [MatCardModule, MatButtonModule, RouterLink],
  templateUrl: './not-found.component.html',
  styleUrl: './not-found.component.scss', 
  encapsulation: ViewEncapsulation.None
})

export class NotFoundComponent {

}
