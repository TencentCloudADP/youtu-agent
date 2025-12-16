import React, { memo } from 'react';

interface BrokenImagePlaceholderProps {
  src: string;
  alt?: string;
  className?: string;
}

const DEFAULT_ALT = 'Image not available';
const DEFAULT_CLASS = '';

const BrokenImagePlaceholder: React.FC<BrokenImagePlaceholderProps> = memo(({
  src,
  alt = DEFAULT_ALT,
  className = DEFAULT_CLASS
}) => (
  <div className={`broken-image-container ${className}`}>
    <div className="broken-image-content">
      <div className="broken-image-icon">
        <i className="fas fa-image"></i>
      </div>
      <div className="broken-image-text">
        <div className="broken-image-alt">{alt}</div>
        <div className="broken-image-url" title={src}>
          {src.length > 50 ? `${src.substring(0, 47)}...` : src}
        </div>
      </div>
    </div>
  </div>
), (prevProps, nextProps) => {
  // Only re-render if src, alt, or className changes
  return prevProps.src === nextProps.src &&
         prevProps.alt === nextProps.alt &&
         prevProps.className === nextProps.className;
});

BrokenImagePlaceholder.displayName = 'BrokenImagePlaceholder';

export default BrokenImagePlaceholder;
