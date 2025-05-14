
import React from 'react';
import ForestMap from '@/components/ForestMap';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MapPin, Info, Compass, Award } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';
import { ForestArea } from '@/components/ForestMap';

const FOREST_AREAS: ForestArea[] = [
  {
    id: 'ancient-grove',
    name: 'Ancient Grove',
    description: 'Old-growth forest with centuries-old trees',
    position: { x: 30, y: 40 },
    tasks: 5,
    completed: 3,
    unlocked: true
  },
  {
    id: 'crystal-lake',
    name: 'Crystal Lake',
    description: 'Serene lake with clear waters and diverse wildlife',
    position: { x: 70, y: 35 },
    tasks: 4,
    completed: 1,
    unlocked: true
  },
  {
    id: 'misty-peaks',
    name: 'Misty Peaks',
    description: 'Foggy mountainous region with unique flora',
    position: { x: 50, y: 20 },
    tasks: 6,
    completed: 0,
    unlocked: false
  },
  {
    id: 'sunlit-meadow',
    name: 'Sunlit Meadow',
    description: 'Open field filled with wildflowers and butterflies',
    position: { x: 20, y: 65 },
    tasks: 3,
    completed: 3,
    unlocked: true
  },
  {
    id: 'hidden-ravine',
    name: 'Hidden Ravine',
    description: 'Mysterious canyon with unexplored caves',
    position: { x: 80, y: 70 },
    tasks: 7,
    completed: 0,
    unlocked: false
  },
];

const Explore: React.FC = () => {
  return (
    <div className="min-h-screen pt-16 pb-24">
      <div className="container mx-auto px-4">
        <div className="py-8">
          <h1 className="text-3xl md:text-4xl font-bold text-forest-primary mb-2">Forest Explorer</h1>
          <p className="text-gray-600 mb-8">
            Discover magical forest areas and complete tasks to unlock new regions
          </p>

          <Tabs defaultValue="map" className="w-full">
            <TabsList className="mb-8">
              <TabsTrigger value="map" className="data-[state=active]:bg-forest-primary data-[state=active]:text-white">
                <MapPin className="h-4 w-4 mr-2" />
                Map View
              </TabsTrigger>
              <TabsTrigger value="list" className="data-[state=active]:bg-forest-primary data-[state=active]:text-white">
                <Info className="h-4 w-4 mr-2" />
                Area Details
              </TabsTrigger>
            </TabsList>

            <TabsContent value="map" className="focus-visible:outline-none focus-visible:ring-0">
              <div className="bg-white rounded-xl shadow-md p-4 md:p-6">
                <ForestMap areas={FOREST_AREAS} />
                
                <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-3 h-3 rounded-full bg-forest-primary"></div>
                    <span>Unlocked Area</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-3 h-3 rounded-full bg-forest-primary/50"></div>
                    <span>Locked Area</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-3 h-3 rounded-full bg-forest-accent"></div>
                    <span>Task Progress</span>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="list" className="focus-visible:outline-none focus-visible:ring-0">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {FOREST_AREAS.map((area) => (
                  <Card key={area.id} className={`forest-card ${!area.unlocked ? 'opacity-75' : ''}`}>
                    <CardHeader>
                      <div className="flex justify-between items-start">
                        <div>
                          <CardTitle className="text-forest-primary">{area.name}</CardTitle>
                          <CardDescription>{area.description}</CardDescription>
                        </div>
                        {area.unlocked ? (
                          <Badge className="bg-forest-primary">Unlocked</Badge>
                        ) : (
                          <Badge variant="outline" className="border-amber-500 text-amber-600">Locked</Badge>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-2 mb-3">
                        <Compass className="h-5 w-5 text-forest-secondary" />
                        <span className="text-sm">Location coordinates: {area.position.x}°N, {area.position.y}°W</span>
                      </div>
                      <div className="flex items-center gap-2 mb-4">
                        <Award className="h-5 w-5 text-forest-secondary" />
                        <span className="text-sm">Tasks: {area.completed || 0}/{area.tasks} completed</span>
                      </div>

                      {/* Progress bar */}
                      <div className="w-full bg-forest-light/30 rounded-full h-2.5 mb-4">
                        <div 
                          className="bg-forest-accent h-2.5 rounded-full transition-all duration-700"
                          style={{ width: `${((area.completed || 0) / area.tasks) * 100}%` }}
                        ></div>
                      </div>

                      {area.unlocked ? (
                        <Link
                          to={`/area/${area.id}`}
                          className="text-forest-primary hover:text-forest-accent font-medium flex items-center"
                        >
                          Explore Area <ArrowRight className="ml-1 h-4 w-4" />
                        </Link>
                      ) : (
                        <div className="text-sm text-gray-500">
                          Complete tasks in other areas to unlock this region
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default Explore;
