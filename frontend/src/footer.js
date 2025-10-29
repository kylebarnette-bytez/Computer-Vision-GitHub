import React from "react";

const Footer = () => {
  const footerStyle = {
    backgroundColor: "#1e628aff",  
    color: "#eff2f8ff",            // text color
    textAlign: "center",
    padding: "2rem",

  };

  return (
    <footer style={footerStyle}>
      &copy; 2025 FoodLens. All rights reserved.
    </footer>
  );
};

export default Footer;
