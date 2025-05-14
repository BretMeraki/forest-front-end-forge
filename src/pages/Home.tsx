
import React from 'react';
import Hero from '@/components/Hero';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Link } from 'react-router-dom';
import { ArrowRight, Map, CheckSquare, Trees, Leaf } from 'lucide-react';

const Home: React.FC = () => {
  const forestFeatures = [
    {
      icon: <Map className="h-8 w-8 text-forest-accent" />,
      title: "Interactive Map",
      description: "Navigate through different areas of the forest with our intuitive map interface."
    },
    {
      icon: <CheckSquare className="h-8 w-8 text-forest-accent" />,
      title: "Forest Tasks",
      description: "Complete various tasks and challenges as you explore the forest environment."
    },
    {
      icon: <Trees className="h-8 w-8 text-forest-accent" />,
      title: "Diverse Environments",
      description: "Discover unique forest regions, each with their own flora, fauna, and challenges."
    },
    {
      icon: <Leaf className="h-8 w-8 text-forest-accent leaf-icon" />,
      title: "Nature Connection",
      description: "Connect with nature through immersive experiences and educational content."
    },
  ];

  return (
    <div className="min-h-screen">
      <Hero
        title="Explore the Enchanted Forest"
        subtitle="Embark on a journey through mystical woods, complete tasks and discover hidden wonders of nature."
        ctaText="Start Exploring"
        ctaLink="/explore"
        backgroundImage="https://images.unsplash.com/photo-1530441353774-54750e501bc3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"
      />

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <h2 className="text-3xl font-bold text-forest-primary text-center mb-12">
            Your Forest Adventure Awaits
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {forestFeatures.map((feature, index) => (
              <Card key={index} className="forest-card border-none">
                <CardContent className="p-6 flex flex-col items-center text-center">
                  <div className="mb-4 p-3 bg-forest-light/20 rounded-full">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-medium text-forest-primary mb-2">{feature.title}</h3>
                  <p className="text-gray-600">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="mt-12 text-center">
            <Button asChild className="forest-button">
              <Link to="/tasks">
                View All Tasks <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4 bg-forest-pattern relative">
        <div className="absolute inset-0 bg-forest-primary/95"></div>
        <div className="container mx-auto relative z-10">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
              Ready to Begin Your Forest Journey?
            </h2>
            <p className="text-xl text-forest-mist/80 mb-8">
              Immerse yourself in the wonders of nature, complete tasks, and uncover the secrets
              of the enchanted forest.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild className="forest-button bg-white text-forest-primary hover:bg-forest-mist">
                <Link to="/explore">
                  Explore the Forest
                </Link>
              </Button>
              <Button asChild className="forest-button bg-transparent border border-white hover:bg-white/10">
                <Link to="/tasks">
                  View Tasks
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
