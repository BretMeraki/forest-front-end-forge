
import React from 'react';
import TaskList from '@/components/TaskList';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Task } from '@/components/TaskCard';

const SAMPLE_TASKS: Task[] = [
  {
    id: 't1',
    title: 'Identify 5 different tree species',
    description: 'Find and photograph five different tree species in the Ancient Grove area.',
    completed: false,
    location: 'Ancient Grove',
    difficulty: 'medium'
  },
  {
    id: 't2',
    title: 'Collect forest samples',
    description: 'Gather leaves, bark, and soil samples for research purposes.',
    completed: true,
    location: 'Ancient Grove',
    difficulty: 'easy'
  },
  {
    id: 't3',
    title: 'Track animal footprints',
    description: 'Find and identify three different animal footprints near the lake.',
    completed: false,
    location: 'Crystal Lake',
    difficulty: 'hard'
  },
  {
    id: 't4',
    title: 'Water quality assessment',
    description: 'Measure the pH levels and clarity of the lake water.',
    completed: false,
    location: 'Crystal Lake',
    difficulty: 'medium'
  },
  {
    id: 't5',
    title: 'Document bird species',
    description: 'Observe and document at least 3 different bird species in the meadow.',
    completed: true,
    location: 'Sunlit Meadow',
    difficulty: 'easy'
  },
  {
    id: 't6',
    title: 'Plant identification challenge',
    description: 'Identify and photograph 10 different wildflower species.',
    completed: true,
    location: 'Sunlit Meadow',
    difficulty: 'medium'
  },
  {
    id: 't7',
    title: 'Collect wildflower seeds',
    description: 'Collect seeds from three different wildflower species for conservation.',
    completed: true,
    location: 'Sunlit Meadow',
    difficulty: 'easy'
  }
];

const Tasks: React.FC = () => {
  return (
    <div className="min-h-screen pt-16 pb-24">
      <div className="container mx-auto px-4">
        <div className="py-8">
          <h1 className="text-3xl md:text-4xl font-bold text-forest-primary mb-2">Forest Tasks</h1>
          <p className="text-gray-600 mb-8">
            Complete these tasks to unlock new areas and progress in your forest journey
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
              <Card className="forest-card">
                <CardHeader>
                  <CardTitle className="text-forest-primary">Your Tasks</CardTitle>
                </CardHeader>
                <CardContent>
                  <TaskList tasks={SAMPLE_TASKS} />
                </CardContent>
              </Card>
            </div>

            <div className="space-y-6">
              <Card className="forest-card">
                <CardHeader>
                  <CardTitle className="text-forest-primary">Task Progress</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Ancient Grove</span>
                        <span className="text-sm font-medium">1/2</span>
                      </div>
                      <div className="w-full bg-forest-light/30 rounded-full h-2">
                        <div className="bg-forest-primary h-2 rounded-full" style={{ width: '50%' }}></div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Crystal Lake</span>
                        <span className="text-sm font-medium">0/2</span>
                      </div>
                      <div className="w-full bg-forest-light/30 rounded-full h-2">
                        <div className="bg-forest-primary h-2 rounded-full" style={{ width: '0%' }}></div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Sunlit Meadow</span>
                        <span className="text-sm font-medium">3/3</span>
                      </div>
                      <div className="w-full bg-forest-light/30 rounded-full h-2">
                        <div className="bg-forest-primary h-2 rounded-full" style={{ width: '100%' }}></div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 pt-6 border-t border-border">
                    <h4 className="font-medium mb-2">Area Unlocking Progress</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Misty Peaks</span>
                        <span>Need 2 more tasks</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Hidden Ravine</span>
                        <span>Need 4 more tasks</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="forest-card">
                <CardHeader>
                  <CardTitle className="text-forest-primary">Task Difficulty Guide</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">easy</span>
                      <span className="text-sm">Beginner-friendly, minimal effort</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs">medium</span>
                      <span className="text-sm">Moderate challenge, some experience needed</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs">hard</span>
                      <span className="text-sm">Advanced tasks, significant effort required</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Tasks;
