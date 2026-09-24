'use client';

import { useState } from 'react';

interface PredictionResult {
  prediction: string;
  confidence: number;
}

interface Hospital {
  name: string;
  location: string;
  specialist: string;
  contact: string;
}

interface ApiResponse {
  status: string;
  patient_id: string;
  patient_age_gender: string;
  scan_modality: string;
  patient_continent: string;
  results: Record<string, PredictionResult>;
  recommended_hospitals: Hospital[];
  image_url: string;
  gradcam_url: string | null;
  pdf_report_url: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState('');
  const [patientAgeGender, setPatientAgeGender] = useState('');
  const [scanModality, setScanModality] = useState('Standard MRI');
  const [patientContinent, setPatientContinent] = useState('Asia');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ApiResponse | null>(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image or DICOM file.');
      return;
    }

    setError('');
    setLoading(true);
    setData(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_id', patientId || 'N/A');
    formData.append('patient_age_gender', patientAgeGender || 'N/A');
    formData.append('scan_modality', scanModality);
    formData.append('patient_continent', patientContinent);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process the scan on the backend server.');
      }

      const result: ApiResponse = await response.json();
      setData(result);
    } catch (err: any) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-12">
      <div className="max-w-5xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="border-b border-slate-800 pb-6">
          <h1 className="text-3xl font-extrabold tracking-tight text-sky-400">NeuroScan AI Dashboard</h1>
          <p className="text-slate-400 mt-1">Multi-Model Ensemble Brain Tumor Classification & Grad-CAM Analysis</p>
        </div>

        {/* Upload Form */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Patient ID</label>
                <input 
                  type="text" 
                  value={patientId} 
                  onChange={(e) => setPatientId(e.target.value)} 
                  placeholder="e.g. P-10042"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-sky-500"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Age / Gender</label>
                <input 
                  type="text" 
                  value={patientAgeGender} 
                  onChange={(e) => setPatientAgeGender(e.target.value)} 
                  placeholder="e.g. 45 / M"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-sky-500"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Modality</label>
                <select 
                  value={scanModality} 
                  onChange={(e) => setScanModality(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-sky-500"
                >
                  <option value="Standard MRI">Standard MRI</option>
                  <option value="DICOM MRI Scan">DICOM MRI Scan</option>
                  <option value="CT Scan">CT Scan</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Patient Region</label>
                <select 
                  value={patientContinent} 
                  onChange={(e) => setPatientContinent(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-sky-500"
                >
                  <option value="Asia">Asia</option>
                  <option value="North America">North America</option>
                  <option value="Europe">Europe</option>
                  <option value="South America">South America</option>
                  <option value="Africa">Africa</option>
                  <option value="Oceania">Oceania</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Upload Scan (.png, .jpg, .dcm)</label>
              <input 
                type="file" 
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="w-full text-sm text-slate-400 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-sky-600 file:text-white hover:file:bg-sky-500 cursor-pointer bg-slate-950 border border-slate-800 rounded-lg"
              />
            </div>

            {error && <p className="text-red-400 text-sm">{error}</p>}

            <button 
              type="submit" 
              disabled={loading}
              className="w-full bg-sky-600 hover:bg-sky-500 text-white font-medium py-3 rounded-lg transition-colors disabled:opacity-50 shadow-lg shadow-sky-900/20"
            >
              {loading ? 'Analyzing Scan with Ensemble Models...' : 'Run Diagnostics'}
            </button>
          </form>
        </div>

        {/* Results Section */}
        {data && (
          <div className="space-y-6 animate-fade-in">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-900 border border-slate-800 rounded-xl p-6">
              <div>
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Uploaded Scan</h3>
                <img src={`http://127.0.0.1:5000${data.image_url}`} alt="Scan" className="rounded-lg border border-slate-800 max-h-64 object-contain mx-auto" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Grad-CAM Heatmap</h3>
                {data.gradcam_url ? (
                  <img src={`http://127.0.0.1:5000${data.gradcam_url}`} alt="Grad-CAM" className="rounded-lg border border-slate-800 max-h-64 object-contain mx-auto" />
                ) : (
                  <p className="text-slate-500 text-sm">Grad-CAM not available</p>
                )}
              </div>
            </div>

            {/* Model Predictions Table */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 overflow-hidden">
              <h3 className="text-lg font-bold mb-4 text-sky-400">Model Predictions</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 bg-slate-950">
                      <th className="p-3">Model Architecture</th>
                      <th className="p-3">Prediction</th>
                      <th className="p-3">Confidence Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.results).map(([modelName, res]) => (
                      <tr key={modelName} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                        <td className="p-3 font-medium text-slate-200">{modelName}</td>
                        <td className="p-3 text-sky-300">{res.prediction}</td>
                        <td className="p-3 text-slate-300">{res.confidence}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-6 flex justify-end">
                <a 
                  href={`http://127.0.0.1:5000${data.pdf_report_url}`} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-6 py-2.5 rounded-lg transition-colors text-sm shadow-lg shadow-emerald-900/20"
                >
                  Download Clinical PDF Report
                </a>
              </div>
            </div>

            {/* Recommended Neuro-Oncology Centers */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <h3 className="text-lg font-bold mb-1 text-sky-400">Verified Neuro-Oncology Centers ({data.patient_continent})</h3>
              <p className="text-xs text-slate-400 mb-4">Top-tier verified specialist facilities matching patient regional profile.</p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {data.recommended_hospitals && data.recommended_hospitals.map((hosp, idx) => (
                  <div key={idx} className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-2.5">
                    <h4 className="font-semibold text-slate-100 text-sm leading-snug">{hosp.name}</h4>
                    <div className="space-y-1 text-xs text-slate-300">
                      <p className="flex items-center gap-1.5"><span className="text-slate-500">📍</span> {hosp.location}</p>
                      <p className="flex items-center gap-1.5"><span className="text-sky-400">👨‍⚕️</span> {hosp.specialist}</p>
                      <p className="flex items-center gap-1.5"><span className="text-slate-500">📞</span> {hosp.contact}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}

      </div>
    </main>
  );
}