
import React, { useState } from 'react';
import TaskCard, { Task } from './TaskCard';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Search } from 'lucide-react';

interface TaskListProps {
  tasks: Task[];
}

const TaskList: React.FC<TaskListProps> = ({ tasks: initialTasks }) => {
  const [tasks, setTasks] = useState<Task[]>(initialTasks);
  const [filter, setFilter] = useState<'all' | 'active' | 'completed'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const handleTaskComplete = (id: string) => {
    setTasks(tasks.map(task => 
      task.id === id ? { ...task, completed: !task.completed } : task
    ));
  };

  const filteredTasks = tasks.filter(task => {
    const matchesFilter = 
      filter === 'all' || 
      (filter === 'active' && !task.completed) ||
      (filter === 'completed' && task.completed);
      
    const matchesSearch = task.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      task.description.toLowerCase().includes(searchQuery.toLowerCase());
      
    return matchesFilter && matchesSearch;
  });

  const completedCount = tasks.filter(task => task.completed).length;
  const totalTasks = tasks.length;
  const progressPercentage = totalTasks > 0 ? (completedCount / totalTasks) * 100 : 0;

  return (
    <div>
      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-forest-primary">
            Progress: {completedCount}/{totalTasks} tasks complete
          </span>
          <span className="text-sm font-medium text-forest-primary">
            {progressPercentage.toFixed(0)}%
          </span>
        </div>
        <div className="w-full bg-forest-light/30 rounded-full h-2.5">
          <div 
            className="bg-forest-accent h-2.5 rounded-full transition-all duration-700 ease-in-out"
            style={{ width: `${progressPercentage}%` }}
          ></div>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="relative flex-grow">
          <Search className="absolute left-3 top-2.5 h-5 w-5 text-muted-foreground" />
          <Input 
            placeholder="Search tasks..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="nature-input pl-10"
          />
        </div>
        
        <div className="flex gap-2">
          <Badge 
            variant={filter === 'all' ? "default" : "outline"}
            className={`cursor-pointer ${filter === 'all' ? 'bg-forest-primary' : ''}`}
            onClick={() => setFilter('all')}
          >
            All
          </Badge>
          <Badge 
            variant={filter === 'active' ? "default" : "outline"}
            className={`cursor-pointer ${filter === 'active' ? 'bg-forest-primary' : ''}`}
            onClick={() => setFilter('active')}
          >
            Active
          </Badge>
          <Badge 
            variant={filter === 'completed' ? "default" : "outline"}
            className={`cursor-pointer ${filter === 'completed' ? 'bg-forest-primary' : ''}`}
            onClick={() => setFilter('completed')}
          >
            Completed
          </Badge>
        </div>
      </div>

      {/* Task List */}
      <div className="space-y-4">
        {filteredTasks.length > 0 ? (
          filteredTasks.map(task => (
            <TaskCard key={task.id} task={task} onTaskComplete={handleTaskComplete} />
          ))
        ) : (
          <div className="text-center py-8">
            <p className="text-forest-primary/70">
              {searchQuery ? "No tasks match your search" : "No tasks available"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TaskList;
