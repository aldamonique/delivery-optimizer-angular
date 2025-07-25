import { TestBed } from '@angular/core/testing';

import { DeliveryPlanService } from './delivery-plan.service';

describe('DeliveryPlanService', () => {
  let service: DeliveryPlanService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(DeliveryPlanService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
