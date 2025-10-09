// K6 Load Test Script for Fraud Detection API

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 10 },  // Ramp up to 10 users
    { duration: '1m', target: 10 },   // Stay at 10 users
    { duration: '30s', target: 20 },  // Ramp up to 20 users
    { duration: '1m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s
    errors: ['rate<0.1'],              // Error rate must be below 10%
  },
};

// Base URL
const BASE_URL = 'http://localhost:8000';

// Sample transaction data
const sampleTransaction = {
  step: 1,
  type: 'TRANSFER',
  amount: 181.0,
  nameOrig: 'C1231006815',
  oldbalanceOrg: 181.0,
  newbalanceOrig: 0.0,
  nameDest: 'C1900366749',
  oldbalanceDest: 0.0,
  newbalanceDest: 0.0
};

export default function () {
  // Test health endpoint
  let healthResponse = http.get(`${BASE_URL}/health`);
  check(healthResponse, {
    'health check status is 200': (r) => r.status === 200,
  }) || errorRate.add(1);

  // Test single prediction
  let predictionResponse = http.post(
    `${BASE_URL}/predict`,
    JSON.stringify(sampleTransaction),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );
  
  check(predictionResponse, {
    'prediction status is 200': (r) => r.status === 200,
    'prediction has fraud_score': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.fraud_score !== undefined;
      } catch (e) {
        return false;
      }
    },
  }) || errorRate.add(1);

  // Test batch prediction (every 5th iteration)
  if (__ITER % 5 === 0) {
    const batchRequest = {
      transactions: [sampleTransaction, sampleTransaction, sampleTransaction]
    };
    
    let batchResponse = http.post(
      `${BASE_URL}/predict/batch`,
      JSON.stringify(batchRequest),
      {
        headers: { 'Content-Type': 'application/json' },
      }
    );
    
    check(batchResponse, {
      'batch prediction status is 200': (r) => r.status === 200,
      'batch prediction has results': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.results && body.results.length === 3;
        } catch (e) {
          return false;
        }
      },
    }) || errorRate.add(1);
  }

  // Test service status (every 10th iteration)
  if (__ITER % 10 === 0) {
    let statusResponse = http.get(`${BASE_URL}/status`);
    check(statusResponse, {
      'status endpoint is 200': (r) => r.status === 200,
    }) || errorRate.add(1);
  }

  sleep(1);
}