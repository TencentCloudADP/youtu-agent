import React from 'react';

interface HamburgerButtonProps {
  isOpen: boolean;
  onClick: () => void;
}

const HamburgerButton: React.FC<HamburgerButtonProps> = ({ isOpen, onClick }) => {
  return (
    <button 
      className="hamburger-button"
      onClick={onClick}
      aria-label="Toggle menu"
      aria-expanded={isOpen}
    >
      <i className={`fas ${isOpen ? 'fa-times' : 'fa-bars'}`}></i>
    </button>
  );
};

export default HamburgerButton;
