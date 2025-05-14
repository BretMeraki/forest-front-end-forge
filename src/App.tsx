
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Explore from './pages/Explore';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Routes>
        <Route path="/" element={<Explore />} />
        <Route path="/explore" element={<Explore />} />
        {/* Add more routes as needed */}
      </Routes>
    </div>
  );
}

export default App;
