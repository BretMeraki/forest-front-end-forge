
import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';

const Explore = () => {
  const journeys = [
    {
      id: 1,
      title: 'Self-Discovery',
      description: 'Explore your values, strengths, and areas for growth',
      path: '/journey/self-discovery'
    },
    {
      id: 2,
      title: 'Emotional Intelligence',
      description: 'Develop awareness and regulation of emotions',
      path: '/journey/emotional-intelligence'
    },
    {
      id: 3,
      title: 'Relationship Building',
      description: 'Strengthen connections and communication skills',
      path: '/journey/relationships'
    },
    {
      id: 4,
      title: 'Purpose & Meaning',
      description: 'Define your vision and align with your values',
      path: '/journey/purpose'
    }
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Explore Growth Journeys</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
        {journeys.map((journey) => (
          <div 
            key={journey.id} 
            className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
          >
            <h2 className="text-xl font-semibold mb-2">{journey.title}</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-4">{journey.description}</p>
            <Link 
              to={journey.path}
              className="flex items-center text-blue-600 dark:text-blue-400 font-medium"
            >
              Begin Journey <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Explore;
