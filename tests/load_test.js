// K6 Load Test Script for Fraud Detection API

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const predictionLatency = new Trend('prediction_latency');
const batchLatency = new Trend('batch_latency');
const explanationLatency = new Trend('explanation_latency');

// Test configuration for comprehensive load testing
export const options = {
  scenarios: {
    // Light load test
    light_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 5 },
        { duration: '2m', target: 5 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '30s',
    },
    
    // Medium load test
    medium_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 20 },
        { duration: '3m', target: 20 },
        { duration: '1m', target: 0 },
      ],
      gracefulRampDown: '30s',
      startTime: '5m',
    },
    
    // Heavy load test
    heavy_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },
        { duration: '5m', target: 50 },
        { duration: '2m', target: 100 },
        { duration: '3m', target: 100 },
        { duration: '2m', target: 0 },
      ],
      gracefulRampDown: '1m',
      startTime: '10m',
    },
    
    // Spike test
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },
        { duration: '30s', target: 200 }, // Sudden spike
        { duration: '1m', target: 200 },
        { duration: '30s', target: 10 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '30s',
      startTime: '25m',
    },
  },
  
  thresholds: {
    // Overall performance thresholds
    http_req_duration: [
      'p(50)<100',    // 50% of requests under 100ms
      'p(95)<500',    // 95% of requests under 500ms
      'p(99)<1000',   // 99% of requests under 1s
    ],
    
    // Custom metric thresholds
    prediction_latency: ['p(95)<200'],     // 95% of predictions under 200ms
    batch_latency: ['p(95)<2000'],         // 95% of batch requests under 2s
    explanation_latency: ['p(95)<1000'],   // 95% of explanations under 1s
    
    // Error rate thresholds
    errors: ['rate<0.05'],                 // Error rate under 5%
    http_req_failed: ['rate<0.05'],        // HTTP failure rate under 5%
    
    // Throughput thresholds
    http_reqs: ['rate>10'],                // At least 10 requests per second
  },
};

// Base URL - can be overridden with environment variable
const BASE_URL = __ENV.API_URL || 'http://localhost:8000';

// Sample transaction data variations for realistic testing
const transactionTemplates = [
  {
    step: 1,
    type: 'TRANSFER',
    amount: 181.0,
    nameOrig: 'C1231006815',
    oldbalanceOrg: 181.0,
    newbalanceOrig: 0.0,
    nameDest: 'C1900366749',
    oldbalanceDest: 0.0,
    newbalanceDest: 0.0
  },
  {
    step: 50,
    type: 'PAYMENT',
    amount: 1500.0,
    nameOrig: 'C2345678901',
    oldbalanceOrg: 5000.0,
    newbalanceOrig: 3500.0,
    nameDest: 'M1234567890',
    oldbalanceDest: 0.0,
    newbalanceDest: 1500.0
  },
  {
    step: 100,
    type: 'CASH-OUT',
    amount: 250000.0,
    nameOrig: 'C3456789012',
    oldbalanceOrg: 300000.0,
    newbalanceOrig: 50000.0,
    nameDest: 'C4567890123',
    oldbalanceDest: 100000.0,
    newbalanceDest: 350000.0
  },
  {
    step: 200,
    type: 'DEBIT',
    amount: 75.0,
    nameOrig: 'C5678901234',
    oldbalanceOrg: 1000.0,
    newbalanceOrig: 925.0,
    nameDest: 'M2345678901',
    oldbalanceDest: 0.0,
    newbalanceDest: 75.0
  },
  {
    step: 300,
    type: 'CASH-IN',
    amount: 10000.0,
    nameOrig: 'C6789012345',
    oldbalanceOrg: 5000.0,
    newbalanceOrig: 15000.0,
    nameDest: 'M3456789012',
    oldbalanceDest: 0.0,
    newbalanceDest: 10000.0
  }
];

// Function to get random transaction
function getRandomTransaction() {
  const template = transactionTemplates[Math.floor(Math.random() * transactionTemplates.length)];
  
  // Add some randomization to make each transaction unique
  return {
    ...template,
    step: template.step + Math.floor(Math.random() * 100),
    amount: template.amount * (0.8 + Math.random() * 0.4), // ±20% variation
    nameOrig: template.nameOrig.slice(0, -3) + String(Math.floor(Math.random() * 1000)).padStart(3, '0'),
    nameDest: template.nameDest.slice(0, -3) + String(Math.floor(Math.random() * 1000)).padStart(3, '0'),
  };
}

export default function () {
  const scenario = __ENV.EXEC_SCENARIO || 'default';
  
  // Vary behavior based on scenario
  switch (scenario) {
    case 'light_load':
      lightLoadTest();
      break;
    case 'medium_load':
      mediumLoadTest();
      break;
    case 'heavy_load':
      heavyLoadTest();
      break;
    case 'spike_test':
      spikeTest();
      break;
    default:
      defaultTest();
  }
}

function defaultTest() {
  // Health check (10% of requests)
  if (Math.random() < 0.1) {
    testHealthEndpoint();
  }
  
  // Single prediction (60% of requests)
  if (Math.random() < 0.6) {
    testSinglePrediction();
  }
  
  // Batch prediction (20% of requests)
  if (Math.random() < 0.2) {
    testBatchPrediction();
  }
  
  // Explanation request (10% of requests)
  if (Math.random() < 0.1) {
    testExplanationRequest();
  }
  
  sleep(Math.random() * 2); // Random sleep 0-2 seconds
}

function lightLoadTest() {
  // Focus on single predictions with minimal load
  testSinglePrediction();
  sleep(1 + Math.random()); // 1-2 second intervals
}

function mediumLoadTest() {
  // Mix of single and batch predictions
  if (Math.random() < 0.7) {
    testSinglePrediction();
  } else {
    testBatchPrediction();
  }
  sleep(0.5 + Math.random() * 0.5); // 0.5-1 second intervals
}

function heavyLoadTest() {
  // High frequency requests with minimal sleep
  const rand = Math.random();
  if (rand < 0.5) {
    testSinglePrediction();
  } else if (rand < 0.8) {
    testBatchPrediction();
  } else {
    testExplanationRequest();
  }
  sleep(0.1 + Math.random() * 0.2); // 0.1-0.3 second intervals
}

function spikeTest() {
  // Rapid-fire requests during spike
  testSinglePrediction();
  if (Math.random() < 0.3) {
    testBatchPrediction();
  }
  sleep(0.05 + Math.random() * 0.1); // Very short intervals
}

function testHealthEndpoint() {
  const response = http.get(`${BASE_URL}/health`);
  
  check(response, {
    'health check status is 200 or 503': (r) => [200, 503].includes(r.status),
  }) || errorRate.add(1);
}

function testSinglePrediction() {
  const transaction = getRandomTransaction();
  const startTime = Date.now();
  
  const response = http.post(
    `${BASE_URL}/predict`,
    JSON.stringify(transaction),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );
  
  const latency = Date.now() - startTime;
  predictionLatency.add(latency);
  
  const success = check(response, {
    'prediction status is 200 or 503': (r) => [200, 503].includes(r.status),
    'prediction response time < 1000ms': (r) => latency < 1000,
    'prediction has fraud_score (if 200)': (r) => {
      if (r.status !== 200) return true; // Skip check if not 200
      try {
        const body = JSON.parse(r.body);
        return body.fraud_score !== undefined && 
               body.risk_level !== undefined &&
               body.is_fraud_prediction !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!success) errorRate.add(1);
}

function testBatchPrediction() {
  const batchSize = 3 + Math.floor(Math.random() * 7); // 3-10 transactions
  const transactions = [];
  
  for (let i = 0; i < batchSize; i++) {
    transactions.push(getRandomTransaction());
  }
  
  const batchRequest = { transactions };
  const startTime = Date.now();
  
  const response = http.post(
    `${BASE_URL}/predict/batch`,
    JSON.stringify(batchRequest),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );
  
  const latency = Date.now() - startTime;
  batchLatency.add(latency);
  
  const success = check(response, {
    'batch prediction status is 200 or 503': (r) => [200, 503].includes(r.status),
    'batch response time reasonable': (r) => latency < 5000, // 5 seconds max
    'batch has correct results (if 200)': (r) => {
      if (r.status !== 200) return true; // Skip check if not 200
      try {
        const body = JSON.parse(r.body);
        return body.results && 
               body.results.length === batchSize &&
               body.total_transactions === batchSize;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!success) errorRate.add(1);
}

function testExplanationRequest() {
  const transaction = getRandomTransaction();
  const startTime = Date.now();
  
  const response = http.post(
    `${BASE_URL}/predict/explain`,
    JSON.stringify(transaction),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );
  
  const latency = Date.now() - startTime;
  explanationLatency.add(latency);
  
  const success = check(response, {
    'explanation status is 200 or 503': (r) => [200, 503].includes(r.status),
    'explanation response time < 2000ms': (r) => latency < 2000,
    'explanation has detailed info (if 200)': (r) => {
      if (r.status !== 200) return true; // Skip check if not 200
      try {
        const body = JSON.parse(r.body);
        return body.fraud_score !== undefined &&
               body.explanation_text !== undefined &&
               body.risk_factors !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!success) errorRate.add(1);
}

function testServiceStatus() {
  const response = http.get(`${BASE_URL}/status`);
  
  check(response, {
    'status endpoint is 200 or 503': (r) => [200, 503].includes(r.status),
  }) || errorRate.add(1);
}

// Export for potential use in other test scenarios
export { testServiceStatus };