
import React, { useState } from 'react';
import { Link } from 'react-router-dom';

export interface ForestArea {
  id: string;
  name: string;
  description: string;
  position: { x: number; y: number };
  tasks: number;
  completed?: number;
  unlocked: boolean;
  image?: string;
}

interface ForestMapProps {
  areas: ForestArea[];
}

const ForestMap: React.FC<ForestMapProps> = ({ areas }) => {
  const [hoveredArea, setHoveredArea] = useState<string | null>(null);

  return (
    <div className="relative w-full aspect-[16/9] rounded-2xl overflow-hidden forest-card">
      {/* Map background */}
      <div className="absolute inset-0 bg-gradient-to-br from-forest-light/10 to-forest-primary/20 backdrop-blur-sm"></div>

      {/* Map grid overlay */}
      <div className="absolute inset-0 grid grid-cols-12 grid-rows-6">
        {Array.from({ length: 72 }).map((_, i) => (
          <div 
            key={i}
            className="border border-forest-primary/5"
          />
        ))}
      </div>

      {/* Forest areas */}
      {areas.map((area) => (
        <div
          key={area.id}
          className={`absolute transition-all duration-300 ${
            !area.unlocked ? 'opacity-50 grayscale' : 
            hoveredArea === area.id ? 'scale-110 z-10' : ''
          }`}
          style={{ 
            left: `${area.position.x}%`, 
            top: `${area.position.y}%`,
            transform: 'translate(-50%, -50%)'
          }}
          onMouseEnter={() => setHoveredArea(area.id)}
          onMouseLeave={() => setHoveredArea(null)}
        >
          <Link 
            to={area.unlocked ? `/area/${area.id}` : '#'}
            className={`relative block w-16 h-16 md:w-20 md:h-20 rounded-full overflow-hidden
              ${area.unlocked ? 'cursor-pointer' : 'cursor-not-allowed'}
              ${hoveredArea === area.id ? 'ring-4 ring-forest-accent shadow-xl' : ''}`}
            onClick={(e) => !area.unlocked && e.preventDefault()}
          >
            <div className="absolute inset-0 bg-forest-primary/50 rounded-full flex items-center justify-center">
              {area.image ? (
                <img 
                  src={area.image} 
                  alt={area.name} 
                  className="w-full h-full object-cover" 
                />
              ) : (
                <div className="w-full h-full bg-forest-primary flex items-center justify-center">
                  <span className="text-xl font-bold text-white">{area.name.charAt(0)}</span>
                </div>
              )}
            </div>
            
            {/* Progress indicator */}
            {area.unlocked && area.tasks > 0 && (
              <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-gray-200">
                <div 
                  className="bg-forest-accent h-full" 
                  style={{ width: `${(area.completed || 0) / area.tasks * 100}%` }}
                ></div>
              </div>
            )}
          </Link>
          
          {/* Area tooltip */}
          {hoveredArea === area.id && (
            <div className="absolute left-1/2 transform -translate-x-1/2 mt-2 z-20
                          bg-white p-3 rounded-lg shadow-lg w-48 text-center">
              <h3 className="font-bold text-forest-primary mb-1">{area.name}</h3>
              <p className="text-xs text-gray-600 mb-1">{area.description}</p>
              {area.unlocked ? (
                <div className="text-xs text-forest-secondary">
                  {area.completed || 0}/{area.tasks} tasks completed
                </div>
              ) : (
                <div className="text-xs text-amber-600 font-medium">Locked Area</div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default ForestMap;
