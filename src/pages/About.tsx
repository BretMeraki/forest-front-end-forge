
import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Leaf, Info, HelpCircle, Shield } from 'lucide-react';

const About: React.FC = () => {
  return (
    <div className="min-h-screen pt-16 pb-24">
      <div className="container mx-auto px-4">
        <div className="py-8">
          <h1 className="text-3xl md:text-4xl font-bold text-forest-primary mb-2">About Forest Explorer</h1>
          <p className="text-gray-600 mb-8">
            Learn more about our immersive forest exploration experience
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            <Card className="forest-card">
              <CardContent className="p-6">
                <div className="flex flex-col md:flex-row md:items-center gap-4">
                  <div className="p-3 bg-forest-light/30 rounded-full">
                    <Info className="h-8 w-8 text-forest-primary" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-forest-primary mb-3">Our Mission</h2>
                    <p className="text-gray-600">
                      Forest Explorer is dedicated to connecting people with nature through immersive
                      experiences that educate, inspire, and foster environmental stewardship. We believe
                      that by engaging with the natural world in a meaningful way, individuals develop a
                      deeper appreciation for forests and their ecological importance.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="forest-card">
              <CardContent className="p-6">
                <div className="flex flex-col md:flex-row md:items-center gap-4">
                  <div className="p-3 bg-forest-light/30 rounded-full">
                    <Leaf className="h-8 w-8 text-forest-primary" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-forest-primary mb-3">Conservation Focus</h2>
                    <p className="text-gray-600">
                      Our platform goes beyond entertainment – it's a tool for conservation education.
                      Through interactive tasks and exploration, users learn about forest ecosystems,
                      biodiversity, and the challenges facing our natural landscapes. A portion of our
                      proceeds supports forest conservation initiatives worldwide.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="mb-12">
            <h2 className="text-2xl font-bold text-forest-primary mb-6">How It Works</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-xl p-6 shadow-sm border border-forest-light/30">
                <div className="w-12 h-12 flex items-center justify-center rounded-full bg-forest-light/30 mb-4">
                  <span className="text-xl font-bold text-forest-primary">1</span>
                </div>
                <h3 className="text-xl font-medium text-forest-primary mb-2">Explore</h3>
                <p className="text-gray-600">
                  Navigate through different forest areas on our interactive map. Each area has
                  unique characteristics and learning opportunities.
                </p>
              </div>

              <div className="bg-white rounded-xl p-6 shadow-sm border border-forest-light/30">
                <div className="w-12 h-12 flex items-center justify-center rounded-full bg-forest-light/30 mb-4">
                  <span className="text-xl font-bold text-forest-primary">2</span>
                </div>
                <h3 className="text-xl font-medium text-forest-primary mb-2">Complete Tasks</h3>
                <p className="text-gray-600">
                  Engage with the environment through educational tasks like species identification,
                  sample collection, and ecosystem observation.
                </p>
              </div>

              <div className="bg-white rounded-xl p-6 shadow-sm border border-forest-light/30">
                <div className="w-12 h-12 flex items-center justify-center rounded-full bg-forest-light/30 mb-4">
                  <span className="text-xl font-bold text-forest-primary">3</span>
                </div>
                <h3 className="text-xl font-medium text-forest-primary mb-2">Unlock New Areas</h3>
                <p className="text-gray-600">
                  As you complete tasks and gain knowledge, you'll unlock new forest regions with
                  more advanced challenges and discoveries.
                </p>
              </div>
            </div>
          </div>

          <div className="mb-12">
            <h2 className="text-2xl font-bold text-forest-primary mb-6">Frequently Asked Questions</h2>
            <div className="space-y-4">
              <FAQItem
                question="Is Forest Explorer suitable for children?"
                answer="Yes! Forest Explorer is designed for nature enthusiasts of all ages. The tasks vary in difficulty, making it accessible for families to explore together."
                icon={<HelpCircle className="h-5 w-5" />}
              />
              <FAQItem
                question="How do I track my progress?"
                answer="Your task completion and area unlocking progress is automatically tracked in your profile and visible on the Tasks page."
                icon={<HelpCircle className="h-5 w-5" />}
              />
              <FAQItem
                question="Is this a real forest I'm exploring?"
                answer="Forest Explorer combines real forest ecology with interactive elements. While the specific map is a virtual representation, the educational content is based on real forest ecosystems."
                icon={<HelpCircle className="h-5 w-5" />}
              />
              <FAQItem
                question="How are new forest areas unlocked?"
                answer="New areas are unlocked by completing a specific number of tasks in the currently accessible areas. The Tasks page shows your progress toward unlocking each area."
                icon={<HelpCircle className="h-5 w-5" />}
              />
            </div>
          </div>

          <Card className="forest-card">
            <CardContent className="p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-forest-light/30 rounded-full flex-shrink-0">
                  <Shield className="h-6 w-6 text-forest-primary" />
                </div>
                <div>
                  <h2 className="text-xl font-medium text-forest-primary mb-2">Our Commitment</h2>
                  <p className="text-gray-600">
                    Forest Explorer is committed to environmental education and conservation. We actively
                    work with forest preservation organizations and incorporate the latest scientific
                    research into our content. Our goal is to inspire a new generation of forest stewards
                    who understand and appreciate these vital ecosystems.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

interface FAQItemProps {
  question: string;
  answer: string;
  icon: React.ReactNode;
}

const FAQItem: React.FC<FAQItemProps> = ({ question, answer, icon }) => {
  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-forest-light/30">
      <div className="flex gap-3">
        <div className="text-forest-primary flex-shrink-0 mt-1">
          {icon}
        </div>
        <div>
          <h3 className="text-lg font-medium text-forest-primary mb-2">{question}</h3>
          <p className="text-gray-600">{answer}</p>
        </div>
      </div>
    </div>
  );
};

export default About;
