const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
    res.json({
        status: "healthy",
        pipeline_loaded: true,
        environment: "development (Node fallback)"
    });
});

app.post('/api/match', (req, res) => {
    const data = req.body;
    
    // Simulate your optimized matching logic response
    const results = [
        {
            entity_id: "S2_MOCK_001",
            business_name: data.business_name,
            business_address: data.business_address,
            country: data.country,
            source: "S2",
            match_decision: true,
            evidence: {
                name_exact: 1,
                addr_exact: 1,
                name_ratio: 1.0,
                addr_ratio: 1.0
            }
        }
    ];

    res.json({
        query: data,
        candidate_count: 1,
        results: results,
        processing_time: 0.12,
        status: "success",
        message: "Simulated python logic response (Node.js fallback backend)"
    });
});

const jobHistory = [];

app.post('/api/batch-match', (req, res) => {
    const job_id = "job_" + Math.floor(Math.random() * 10000);
    const timestamp = new Date().toISOString();
    
    // Simulate batch processing
    const results = [
        {
            entity_id: "S2_" + Math.floor(Math.random() * 9999),
            business_name: "Simulated Match Corp",
            business_address: "123 Mock St",
            country: "US",
            source: "S2",
            match_decision: true,
            evidence: { name_ratio: 0.95, addr_ratio: 0.95, name_exact: 1 }
        },
        {
            entity_id: "S3_" + Math.floor(Math.random() * 9999),
            business_name: "Simulated Mismatch Inc",
            business_address: "999 No Match Ln",
            country: "UK",
            source: "S3",
            match_decision: false,
            evidence: { name_ratio: 0.4, addr_ratio: 0.2, name_exact: 0 }
        },
        {
            entity_id: "S2_" + Math.floor(Math.random() * 9999),
            business_name: "High Confidence Match",
            business_address: "410 Terry Ave",
            country: "US",
            source: "S2",
            match_decision: true,
            evidence: { name_ratio: 0.98, addr_ratio: 0.99, name_exact: 1 }
        }
    ];

    const jobRecord = {
        id: job_id,
        date: timestamp,
        total_processed: 154,
        matches_found: 2,
        mismatches: 1,
        results: results
    };
    jobHistory.unshift(jobRecord);

    res.json({
        job_id: job_id,
        status: "completed",
        results: results,
        message: "Simulated batch processing against existing sources"
    });
});

app.get('/api/history', (req, res) => {
    res.json(jobHistory);
});

app.listen(8000, () => {
    console.log("====================================");
    console.log("BACKEND SERVER RUNNING AT: http://localhost:8000");
    console.log("====================================");
});
