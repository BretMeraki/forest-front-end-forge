
import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Map, CheckCircle, Circle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Task } from '@/components/TaskCard';

// Mock data for area details - in a real app, this would come from an API or database
const AREA_DETAILS: Record<string, {
  name: string;
  description: string;
  longDescription: string;
  image: string;
  tasks: Task[];
  facts: { title: string; description: string }[];
}> = {
  'ancient-grove': {
    name: 'Ancient Grove',
    description: 'Old-growth forest with centuries-old trees',
    longDescription: 'The Ancient Grove is a remarkable old-growth forest featuring trees that have stood for centuries. Walking among these giants offers a glimpse into the past, as some trees date back over 500 years. The forest floor is carpeted with moss and ferns, while sunlight filters through the dense canopy creating magical light patterns.',
    image: 'https://images.unsplash.com/photo-1473448912268-2022ce9509d8?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80',
    tasks: [
      {
        id: 't1',
        title: 'Identify 5 different tree species',
        description: 'Find and photograph five different tree species in the Ancient Grove area.',
        completed: false,
        difficulty: 'medium'
      },
      {
        id: 't2',
        title: 'Collect forest samples',
        description: 'Gather leaves, bark, and soil samples for research purposes.',
        completed: true,
        difficulty: 'easy'
      },
      {
        id: 't8',
        title: 'Measure tree circumference',
        description: 'Measure and record the circumference of three ancient trees.',
        completed: false,
        difficulty: 'easy'
      },
      {
        id: 't9',
        title: 'Document forest undergrowth',
        description: 'Identify and document five different species in the forest undergrowth.',
        completed: false,
        difficulty: 'medium'
      },
      {
        id: 't10',
        title: 'Find evidence of wildlife',
        description: 'Locate and document three signs of wildlife activity in the grove.',
        completed: true,
        difficulty: 'hard'
      }
    ],
    facts: [
      { 
        title: 'Ancient Ecosystem', 
        description: 'The Ancient Grove contains trees that are over 500 years old, providing habitat for specialized species found nowhere else in the region.' 
      },
      { 
        title: 'Carbon Storage', 
        description: 'Old-growth forests like this one store significantly more carbon than younger forests, making them crucial in the fight against climate change.' 
      },
      { 
        title: 'Biodiversity Hotspot', 
        description: 'Despite covering a relatively small area, the Ancient Grove hosts over 200 plant species and dozens of vertebrate animals.' 
      }
    ]
  },
  'crystal-lake': {
    name: 'Crystal Lake',
    description: 'Serene lake with clear waters and diverse wildlife',
    longDescription: 'Crystal Lake is renowned for its pristine, clear waters that reflect the surrounding forest like a mirror. The lake ecosystem supports a diverse array of aquatic life, from fish to amphibians. The shoreline transitions between rocky outcroppings and sandy beaches, while waterfowl can often be spotted gliding across the surface.',
    image: 'https://images.unsplash.com/photo-1500829243541-74b677fecc30?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80',
    tasks: [
      {
        id: 't3',
        title: 'Track animal footprints',
        description: 'Find and identify three different animal footprints near the lake.',
        completed: false,
        difficulty: 'hard'
      },
      {
        id: 't4',
        title: 'Water quality assessment',
        description: 'Measure the pH levels and clarity of the lake water.',
        completed: false,
        difficulty: 'medium'
      },
      {
        id: 't11',
        title: 'Aquatic plant survey',
        description: 'Identify and document four different aquatic plant species.',
        completed: false,
        difficulty: 'medium'
      },
      {
        id: 't12',
        title: 'Bird watching challenge',
        description: 'Spot and identify five waterbird species that inhabit the lake area.',
        completed: false,
        difficulty: 'medium'
      }
    ],
    facts: [
      { 
        title: 'Glacial Formation', 
        description: 'Crystal Lake was formed approximately 12,000 years ago during the last ice age by retreating glaciers.' 
      },
      { 
        title: 'Water Clarity', 
        description: 'The lake\'s exceptional clarity allows sunlight to penetrate up to 30 feet deep, supporting underwater plant life.' 
      },
      { 
        title: 'Protected Wetlands', 
        description: 'The northern shore of Crystal Lake includes protected wetlands that serve as crucial breeding grounds for amphibians.' 
      }
    ]
  },
  'sunlit-meadow': {
    name: 'Sunlit Meadow',
    description: 'Open field filled with wildflowers and butterflies',
    longDescription: 'Sunlit Meadow is a vibrant open space teeming with colorful wildflowers that bloom in waves throughout the growing season. This meadow ecosystem is particularly known for its butterfly population, with dozens of species visiting to feed on nectar. The area is surrounded by a mix of deciduous trees that provide a stunning backdrop, especially during autumn.',
    image: 'https://images.unsplash.com/photo-1502082553048-f009c37129b9?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80',
    tasks: [
      {
        id: 't5',
        title: 'Document bird species',
        description: 'Observe and document at least 3 different bird species in the meadow.',
        completed: true,
        difficulty: 'easy'
      },
      {
        id: 't6',
        title: 'Plant identification challenge',
        description: 'Identify and photograph 10 different wildflower species.',
        completed: true,
        difficulty: 'medium'
      },
      {
        id: 't7',
        title: 'Collect wildflower seeds',
        description: 'Collect seeds from three different wildflower species for conservation.',
        completed: true,
        difficulty: 'easy'
      }
    ],
    facts: [
      { 
        title: 'Pollinator Paradise', 
        description: 'Sunlit Meadow supports over 30 species of butterflies and numerous bee species, making it essential for regional pollination.' 
      },
      { 
        title: 'Seasonal Blooms', 
        description: 'The meadow displays different dominant flowers each month, creating an ever-changing palette of colors from spring through fall.' 
      },
      { 
        title: 'Restoration Success', 
        description: 'Once agricultural land, this meadow was restored to native wildflowers and grasses through a community conservation effort.' 
      }
    ]
  }
};

const AreaDetail: React.FC = () => {
  const { areaId } = useParams<{ areaId: string }>();
  const areaData = areaId && AREA_DETAILS[areaId];
  
  // Calculate completion stats
  const totalTasks = areaData?.tasks.length || 0;
  const completedTasks = areaData?.tasks.filter(t => t.completed).length || 0;
  const completionPercentage = totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0;
  
  if (!areaData) {
    return (
      <div className="min-h-screen pt-20 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-forest-primary mb-4">Area Not Found</h1>
          <p className="mb-6">Sorry, the forest area you're looking for doesn't exist.</p>
          <Button asChild className="forest-button">
            <Link to="/explore">
              <ArrowLeft className="mr-2 h-5 w-5" />
              Return to Map
            </Link>
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-16 pb-24">
      {/* Hero Banner */}
      <div 
        className="relative h-64 md:h-80 lg:h-96 w-full bg-cover bg-center"
        style={{ backgroundImage: `url(${areaData.image})` }}
      >
        <div className="absolute inset-0 bg-gradient-to-b from-black/50 to-transparent"></div>
        <div className="absolute inset-0 flex flex-col justify-center px-4 md:px-10">
          <div className="container mx-auto">
            <Button 
              asChild 
              variant="outline" 
              className="mb-4 bg-white/80 hover:bg-white text-forest-primary"
            >
              <Link to="/explore">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Map
              </Link>
            </Button>
            <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">{areaData.name}</h1>
            <p className="text-white/90 max-w-2xl">{areaData.description}</p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-wrap items-center gap-3 mb-8">
          <Badge className="bg-forest-primary">{completedTasks}/{totalTasks} Tasks Completed</Badge>
          <div className="flex items-center gap-2 text-sm">
            <span>Completion:</span>
            <div className="w-32 bg-forest-light/30 rounded-full h-2">
              <div 
                className="bg-forest-accent h-2 rounded-full transition-all duration-700"
                style={{ width: `${completionPercentage}%` }}
              ></div>
            </div>
            <span>{completionPercentage.toFixed(0)}%</span>
          </div>
        </div>

        <Tabs defaultValue="details" className="w-full">
          <TabsList className="mb-8">
            <TabsTrigger value="details" className="data-[state=active]:bg-forest-primary data-[state=active]:text-white">
              Details
            </TabsTrigger>
            <TabsTrigger value="tasks" className="data-[state=active]:bg-forest-primary data-[state=active]:text-white">
              Area Tasks
            </TabsTrigger>
            <TabsTrigger value="facts" className="data-[state=active]:bg-forest-primary data-[state=active]:text-white">
              Interesting Facts
            </TabsTrigger>
          </TabsList>

          <TabsContent value="details" className="focus-visible:outline-none focus-visible:ring-0">
            <Card className="forest-card">
              <CardContent className="p-6">
                <div className="prose max-w-none">
                  <h2 className="text-2xl font-bold text-forest-primary mb-4">About {areaData.name}</h2>
                  <p className="text-gray-700 mb-6">{areaData.longDescription}</p>
                  
                  <div className="my-8">
                    <h3 className="text-xl font-medium text-forest-primary mb-4">Area Map</h3>
                    <div className="bg-forest-mist/50 rounded-lg p-6 flex items-center justify-center">
                      <div className="flex flex-col items-center">
                        <Map className="h-12 w-12 text-forest-primary mb-3" />
                        <p className="text-forest-primary font-medium">Interactive area map coming soon!</p>
                        <p className="text-sm text-gray-600">Explore this region in detail with our upcoming interactive map feature.</p>
                      </div>
                    </div>
                  </div>
                  
                  <h3 className="text-xl font-medium text-forest-primary mb-4">Task Progress</h3>
                  <div className="mb-8">
                    <div className="flex justify-between mb-2">
                      <span>Task Completion</span>
                      <span>{completedTasks}/{totalTasks} Tasks ({completionPercentage.toFixed(0)}%)</span>
                    </div>
                    <div className="w-full bg-forest-light/30 rounded-full h-3">
                      <div 
                        className="bg-forest-accent h-3 rounded-full transition-all duration-700"
                        style={{ width: `${completionPercentage}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <Button asChild className="forest-button">
                    <Link to="/tasks">View All Forest Tasks</Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="tasks" className="focus-visible:outline-none focus-visible:ring-0">
            <Card className="forest-card">
              <CardHeader>
                <CardTitle className="text-forest-primary">Tasks in {areaData.name}</CardTitle>
              </CardHeader>
              <CardContent>
                {areaData.tasks.map((task) => (
                  <div 
                    key={task.id} 
                    className={`mb-4 p-4 bg-white rounded-lg border ${task.completed ? 'border-forest-primary/30' : 'border-gray-200'}`}
                  >
                    <div className="flex gap-3">
                      <div className="mt-1 flex-shrink-0">
                        {task.completed ? (
                          <CheckCircle className="h-5 w-5 text-forest-primary" />
                        ) : (
                          <Circle className="h-5 w-5 text-gray-400" />
                        )}
                      </div>
                      <div>
                        <div className="flex items-center justify-between">
                          <h3 className={`font-medium ${task.completed ? 'text-forest-primary/80 line-through' : 'text-forest-primary'}`}>
                            {task.title}
                          </h3>
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            task.difficulty === 'easy' ? 'bg-green-100 text-green-800' :
                            task.difficulty === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {task.difficulty}
                          </span>
                        </div>
                        <p className="text-gray-600 text-sm mt-1">{task.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="facts" className="focus-visible:outline-none focus-visible:ring-0">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {areaData.facts.map((fact, index) => (
                <Card key={index} className="forest-card">
                  <CardContent className="p-6">
                    <h3 className="text-lg font-medium text-forest-primary mb-2">{fact.title}</h3>
                    <p className="text-gray-600 text-sm">{fact.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default AreaDetail;
