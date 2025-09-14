import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Upload,
  Settings,
  Activity,
  TrendingUp,
  Moon,
  Sun,
  LineChart,
  Target
} from 'lucide-react';
import { Button } from '../ui/button';

interface SidebarProps {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

const navigation = [
  { name: '전략적 대시보드', href: '/strategic', icon: Target },
  { name: 'Data 업로드', href: '/data', icon: Upload },
  { name: '모델링', href: '/modeling', icon: Settings },
  { name: 'Dashboard', href: '/dashboard', icon: Activity },
  { name: '기대효과', href: '/effects', icon: TrendingUp },
];

export const Sidebar: React.FC<SidebarProps> = ({ isDarkMode, toggleDarkMode }) => {
  return (
    <div className="w-64 bg-background border-r border-border h-screen flex flex-col flex-shrink-0">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <h1 className="text-xl font-bold text-foreground">Salary Increase Forecast</h1>
        <p className="text-sm text-muted-foreground">임금인상률 예측 시스템</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              `flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              }`
            }
          >
            <item.icon className="mr-3 h-5 w-5" />
            {item.name}
          </NavLink>
        ))}
      </nav>

      {/* ExplainerDashboard Link */}
      <div className="px-4 pb-2">
        <NavLink
          to="/explainer"
          className={({ isActive }) =>
            `flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              isActive
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
            }`
          }
        >
          <LineChart className="mr-3 h-5 w-5" />
          Explainer Dashboard
        </NavLink>
      </div>

      {/* Theme Toggle */}
      <div className="p-4 border-t border-border">
        <Button
          variant="outline"
          size="sm"
          onClick={toggleDarkMode}
          className="w-full"
        >
          {isDarkMode ? (
            <>
              <Sun className="mr-2 h-4 w-4" />
              Light Mode
            </>
          ) : (
            <>
              <Moon className="mr-2 h-4 w-4" />
              Dark Mode
            </>
          )}
        </Button>
      </div>
    </div>
  );
};