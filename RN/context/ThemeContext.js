// context/ThemeContext.js
import React from 'react';

export const ThemeContext = React.createContext({
  isDarkTheme: false,
  toggleTheme: () => {},
});

export const ThemeProvider = ({ children }) => {
  const [isDarkTheme, setIsDarkTheme] = React.useState(false);

  const toggleTheme = React.useCallback(() => {
    setIsDarkTheme(prevTheme => !prevTheme);
  }, []);

  return (
    <ThemeContext.Provider value={{ isDarkTheme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};