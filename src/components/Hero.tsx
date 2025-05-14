
import React from 'react';
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';

interface HeroProps {
  title: string;
  subtitle: string;
  ctaText: string;
  ctaLink: string;
  backgroundImage?: string;
}

const Hero: React.FC<HeroProps> = ({ 
  title, 
  subtitle, 
  ctaText, 
  ctaLink,
  backgroundImage 
}) => {
  return (
    <div className="relative min-h-[90vh] flex items-center overflow-hidden">
      {/* Background */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat" 
        style={{ 
          backgroundImage: backgroundImage ? `url(${backgroundImage})` : 'var(--forest-gradient)',
          backgroundPosition: 'center',
        }}
      >
        {/* Overlay */}
        <div className="absolute inset-0 bg-forest-canopy/60 backdrop-filter backdrop-blur-[2px]"></div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 relative z-10 py-20">
        <div className="max-w-2xl">
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-4 animate-fade-in" style={{ animationDelay: '0.2s' }}>
            {title}
          </h1>
          <p className="text-xl md:text-2xl text-forest-mist mb-8 animate-fade-in" style={{ animationDelay: '0.4s' }}>
            {subtitle}
          </p>
          <Button 
            asChild
            className="forest-button text-lg group animate-fade-in" 
            style={{ animationDelay: '0.6s' }}
          >
            <Link to={ctaLink}>
              {ctaText} 
              <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Link>
          </Button>
        </div>
      </div>

      {/* Decorative Elements */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent"></div>
    </div>
  );
};

export default Hero;
