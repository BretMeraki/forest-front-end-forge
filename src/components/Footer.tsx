
import React from 'react';
import { Link } from 'react-router-dom';

const Footer: React.FC = () => {
  return (
    <footer className="bg-forest-canopy text-white py-12">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Logo and Description */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <img src="/src/assets/forest-logo.svg" alt="Forest Explorer" className="h-10 w-10" />
              <span className="font-bold text-xl">Forest Explorer</span>
            </div>
            <p className="text-forest-mist/80 max-w-md">
              Embark on a journey through enchanted woods, complete tasks, and discover the 
              wonders of nature in our immersive forest experience.
            </p>
          </div>

          {/* Links */}
          <div>
            <h3 className="font-medium text-lg mb-4 text-forest-accent">Navigation</h3>
            <ul className="space-y-2">
              <li><Link to="/" className="text-forest-mist/80 hover:text-white transition-colors">Home</Link></li>
              <li><Link to="/explore" className="text-forest-mist/80 hover:text-white transition-colors">Explore</Link></li>
              <li><Link to="/tasks" className="text-forest-mist/80 hover:text-white transition-colors">Tasks</Link></li>
              <li><Link to="/about" className="text-forest-mist/80 hover:text-white transition-colors">About</Link></li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="font-medium text-lg mb-4 text-forest-accent">Connect</h3>
            <ul className="space-y-2">
              <li><a href="#" className="text-forest-mist/80 hover:text-white transition-colors">Contact Us</a></li>
              <li><a href="#" className="text-forest-mist/80 hover:text-white transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="text-forest-mist/80 hover:text-white transition-colors">Terms of Service</a></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-forest-mist/20 mt-8 pt-8 text-center text-forest-mist/60 text-sm">
          <p>&copy; {new Date().getFullYear()} Forest Explorer. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
