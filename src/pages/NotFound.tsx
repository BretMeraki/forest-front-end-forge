
import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-forest-mist/50 px-4">
      <div className="forest-card p-8 max-w-md w-full text-center">
        <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-forest-light/50 flex items-center justify-center">
          <span className="text-4xl font-bold text-forest-primary">404</span>
        </div>
        <h1 className="text-2xl font-bold text-forest-primary mb-2">Path Not Found</h1>
        <p className="text-gray-600 mb-6">
          Sorry, the forest path you're looking for doesn't exist. Let's guide you back to a known trail.
        </p>
        <Button asChild className="forest-button w-full">
          <Link to="/">
            <ArrowLeft className="mr-2 h-5 w-5" />
            Return to Forest Entrance
          </Link>
        </Button>
      </div>
    </div>
  );
};

export default NotFound;
