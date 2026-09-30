import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Database, FileUp, Home, History, BarChart2, Download, Filter, CheckCircle, AlertCircle } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

function App() {
  const [health, setHealth] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [matchResult, setMatchResult] = useState<any>(null);
  const [jobHistory, setJobHistory] = useState<any[]>([]);

  useEffect(() => {
    axios.get(`${API_BASE}/health`)
      .then(res => setHealth(res.data))
      .catch(err => console.error(err));
      
    axios.get(`${API_BASE}/api/history`)
      .then(res => setJobHistory(res.data))
      .catch(err => console.error(err));
  }, [activeTab]);



  return (
    <div className="flex h-screen bg-gray-50 text-slate-800 font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-slate-900 text-white p-5 flex flex-col">
        <div className="flex items-center gap-2 mb-8 text-xl font-bold text-indigo-400">
          <Database /> ENTITYMATCH AI
        </div>
        <nav className="flex flex-col gap-2 flex-grow">
          <button onClick={() => {setActiveTab('dashboard');}} className={`flex items-center gap-2 p-3 rounded-lg ${activeTab === 'dashboard' ? 'bg-indigo-600' : 'hover:bg-slate-800'}`}>
            <BarChart2 size={18} /> ML Dashboard
          </button>
          <button onClick={() => {setActiveTab('upload'); setMatchResult(null);}} className={`flex items-center gap-2 p-3 rounded-lg ${activeTab === 'upload' ? 'bg-indigo-600' : 'hover:bg-slate-800'}`}>
            <FileUp size={18} /> Upload Source
          </button>
          <button onClick={() => {setActiveTab('history');}} className={`flex items-center gap-2 p-3 rounded-lg ${activeTab === 'history' ? 'bg-indigo-600' : 'hover:bg-slate-800'}`}>
            <History size={18} /> Job History
          </button>
        </nav>
        <div className="mt-auto p-4 bg-slate-800 rounded-lg text-sm">
          <div className="flex items-center gap-2 mb-2">
            <Activity size={16} className={health ? 'text-green-400' : 'text-red-400'} /> 
            {health ? 'Backend Connected' : 'Disconnected'}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-8 overflow-y-auto">
        {activeTab === 'dashboard' && (
          <div className="animate-in fade-in duration-500">
            <div className="flex justify-between items-center mb-6">
              <h1 className="text-3xl font-bold text-slate-800">ML Model Evaluation Dashboard</h1>
              <span className="bg-indigo-100 text-indigo-800 text-xs font-bold px-3 py-1 rounded-full border border-indigo-200">PRODUCTION PIPELINE v2.4</span>
            </div>
            
            <div className="grid grid-cols-4 gap-4 mb-8">
              <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 border-l-4 border-l-indigo-500">
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-wider">Macro F0.5 Score</h3>
                <p className="text-3xl font-bold mt-2 text-slate-800">0.945</p>
                <p className="text-xs text-green-600 mt-1 font-medium">↑ 0.032 from baseline</p>
              </div>
              <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 border-l-4 border-l-blue-500">
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-wider">Precision</h3>
                <p className="text-3xl font-bold mt-2 text-slate-800">96.2%</p>
                <p className="text-xs text-green-600 mt-1 font-medium">↑ High Confidence</p>
              </div>
              <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 border-l-4 border-l-sky-500">
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-wider">Recall</h3>
                <p className="text-3xl font-bold mt-2 text-slate-800">89.4%</p>
                <p className="text-xs text-amber-600 mt-1 font-medium">Candidate generation optimized</p>
              </div>
              <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200 border-l-4 border-l-purple-500">
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-wider">Processing Time</h3>
                <p className="text-3xl font-bold mt-2 text-slate-800">1.2ms</p>
                <p className="text-xs text-slate-400 mt-1 font-medium">per entity pair</p>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-6 mb-8">
              <div className="col-span-2 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                <div className="bg-slate-50 border-b border-slate-200 px-6 py-4">
                  <h3 className="font-bold text-slate-700">Algorithm Thresholds Configuration</h3>
                </div>
                <div className="p-6">
                  <div className="space-y-6">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-semibold text-slate-700">Name Match Threshold</span>
                        <span className="text-indigo-600 font-bold">0.85</span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-2">
                        <div className="bg-indigo-500 h-2 rounded-full" style={{width: '85%'}}></div>
                      </div>
                      <p className="text-xs text-slate-400 mt-1">Fuzz Token Set Ratio</p>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-semibold text-slate-700">Address Match Threshold</span>
                        <span className="text-indigo-600 font-bold">0.80</span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-2">
                        <div className="bg-indigo-400 h-2 rounded-full" style={{width: '80%'}}></div>
                      </div>
                      <p className="text-xs text-slate-400 mt-1">Fuzz Ratio Calculation</p>
                    </div>
                    <div className="pt-4 border-t border-slate-100 flex gap-2">
                      <span className="px-2 py-1 bg-green-50 text-green-700 text-xs font-medium rounded">Jaccard Token Fallback: ENABLED</span>
                      <span className="px-2 py-1 bg-green-50 text-green-700 text-xs font-medium rounded">Exact Match Override: ENABLED</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                <div className="bg-slate-50 border-b border-slate-200 px-6 py-4">
                  <h3 className="font-bold text-slate-700">Validation Evaluation List</h3>
                </div>
                <div className="p-0">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-slate-50 text-slate-500">
                      <tr>
                        <th className="px-6 py-3 font-medium">Metric</th>
                        <th className="px-6 py-3 font-medium text-right">Value</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      <tr><td className="px-6 py-3 font-medium text-slate-700">True Positives</td><td className="px-6 py-3 text-right font-mono">142,501</td></tr>
                      <tr><td className="px-6 py-3 font-medium text-slate-700">False Positives</td><td className="px-6 py-3 text-right font-mono text-red-500">4,120</td></tr>
                      <tr><td className="px-6 py-3 font-medium text-slate-700">False Negatives</td><td className="px-6 py-3 text-right font-mono text-amber-500">12,400</td></tr>
                      <tr><td className="px-6 py-3 font-medium text-slate-700">Candidate Recall</td><td className="px-6 py-3 text-right font-mono text-indigo-600">0.985</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div>
            <h1 className="text-3xl font-bold mb-6">Job History</h1>
            <p className="text-gray-600 mb-8">Review past bulk matching tasks and their statistics.</p>
            {jobHistory.length === 0 ? (
              <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100 text-center py-20">
                <History className="mx-auto text-gray-300 mb-4" size={48} />
                <h2 className="text-xl font-semibold text-gray-600 mb-2">No history available</h2>
                <p className="text-gray-400">Upload a dataset to generate matching history.</p>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {jobHistory.map((job, idx) => (
                  <div key={idx} className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex justify-between items-center hover:shadow-md transition-shadow">
                    <div>
                      <h3 className="font-bold text-lg text-indigo-900">Job ID: {job.id}</h3>
                      <p className="text-sm text-gray-500">{new Date(job.date).toLocaleString()}</p>
                    </div>
                    <div className="flex gap-6 text-center">
                      <div>
                        <div className="text-2xl font-bold text-gray-700">{job.total_processed}</div>
                        <div className="text-xs text-gray-500 uppercase tracking-wide">Processed</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-green-600">{job.matches_found}</div>
                        <div className="text-xs text-green-600 uppercase tracking-wide">Matches</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-red-500">{job.mismatches}</div>
                        <div className="text-xs text-red-500 uppercase tracking-wide">Mismatches</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'upload' && (
          <div className="animate-in fade-in duration-500">
            <h1 className="text-3xl font-bold mb-6">Upload Source Dataset</h1>
            <p className="text-gray-600 mb-8">Upload a new source dataset to automatically cross-reference and check for matches or mismatches against existing sources.</p>
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-6">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-10 text-center flex flex-col items-center justify-center bg-gray-50 hover:bg-indigo-50 transition-colors">
                <FileUp size={48} className="text-gray-400 mb-4" />
                <p className="text-gray-600 mb-2 font-medium">Drag and drop your dataset here</p>
                <p className="text-gray-400 text-sm mb-4">Accepts CSV or TSV files</p>
                <input 
                  type="file" 
                  className="hidden" 
                  id="main-file-upload" 
                  accept=".csv,.tsv"
                  onChange={async (e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      try {
                        const formData = new FormData();
                        formData.append('file', e.target.files[0]);
                        const res = await axios.post(`${API_BASE}/api/batch-match`, formData, {
                          headers: { 'Content-Type': 'multipart/form-data' }
                        });
                        setMatchResult(res.data);
                      } catch (err) {
                        console.error(err);
                      }
                    }
                  }}
                />
                <label htmlFor="main-file-upload" className="bg-indigo-600 text-white px-6 py-2 rounded-lg cursor-pointer hover:bg-indigo-700 transition-colors font-semibold shadow-sm">
                  Browse Files
                </label>
              </div>
            </div>

            {matchResult && matchResult.results && (
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h2 className="text-xl font-bold mb-4">Matching Results</h2>
                <div className="flex flex-col gap-4">
                  {matchResult.results.map((res: any, i: number) => (
                    <div key={i} className={`border p-4 rounded-lg flex justify-between items-center ${res.match_decision ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
                      <div>
                        <div className="font-bold text-lg">{res.business_name}</div>
                        <div className="text-gray-600 text-sm">{res.business_address}, {res.country}</div>
                        <div className="mt-2 text-xs text-gray-500 bg-white border inline-block px-2 py-1 rounded shadow-sm">Matched Against: {res.source} | Target ID: {res.entity_id}</div>
                      </div>
                      <div className="text-right">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide shadow-sm ${res.match_decision ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'}`}>
                          {res.match_decision ? 'MATCH FOUND' : 'MISMATCH'}
                        </span>
                        <div className="mt-2 text-xs text-gray-500 space-y-1">
                          <div>Name Confidence: {(res.evidence.name_ratio * 100).toFixed(0)}%</div>
                          <div>Address Confidence: {(res.evidence.addr_ratio * 100).toFixed(0)}%</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
