
import React from 'react';
import { CheckCircle, Circle } from 'lucide-react';

export interface Task {
  id: string;
  title: string;
  description: string;
  completed: boolean;
  location?: string;
  difficulty?: 'easy' | 'medium' | 'hard';
}

interface TaskCardProps {
  task: Task;
  onTaskComplete: (id: string) => void;
}

const TaskCard: React.FC<TaskCardProps> = ({ task, onTaskComplete }) => {
  const getDifficultyColor = () => {
    switch (task.difficulty) {
      case 'easy':
        return 'bg-green-100 text-green-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'hard':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className={`forest-card p-5 transition-all duration-300 ${task.completed ? 'opacity-75' : ''}`}>
      <div className="flex items-start gap-3">
        <button
          onClick={() => onTaskComplete(task.id)}
          className="mt-1 flex-shrink-0 text-forest-primary hover:text-forest-accent transition-colors"
        >
          {task.completed ? (
            <CheckCircle className="h-6 w-6" />
          ) : (
            <Circle className="h-6 w-6" />
          )}
        </button>

        <div className="flex-grow">
          <div className="flex items-center justify-between mb-1">
            <h3 className={`font-medium text-lg ${task.completed ? 'line-through text-forest-primary/70' : 'text-forest-primary'}`}>
              {task.title}
            </h3>
            
            {task.difficulty && (
              <span className={`text-xs px-2 py-1 rounded-full ${getDifficultyColor()}`}>
                {task.difficulty}
              </span>
            )}
          </div>

          <p className="text-gray-600 text-sm mb-3">{task.description}</p>
          
          {task.location && (
            <div className="text-sm text-forest-secondary font-medium">
              Location: {task.location}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TaskCard;
