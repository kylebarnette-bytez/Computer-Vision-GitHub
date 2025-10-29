import { useState } from "react";
import logo from "./logo.png";

const Navbar = () => {
  const [hovered, setHovered] = useState(null);

    const navbarStyle ={

        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "0.5rem 1.5rem",
        backgroundColor: "#1e628aff",
        color: "white",
        position: "sticky",
        top: 0,
        zIndex: 1000,
        fontFamily: "Ariel, sans-serif",
        height: "50px",
    };

    const logoStyle = {
        height: "80px",
        width: "auto",
        cursor: "pointer",
        marginTop: "10px",
        paddingTop: "50px",
        marginLeft: "15px",
      
    };

    const navLinkStyle ={
        liststyle: "none",
        display: "flex",
        gap: "2rem",
        margin: 0,
        padding: 0,
        alignItems: "center",
    }
    const LinkStyle ={

        color: "white",
        textDecoration: "none",
        fontWeight: "500",
    }

    const navItemStyle = {
        margin: 0,
        padding: 0,

    };

    const linkHoverStyle = {
        color: "#bdbb1fff", 
    };


    return (
     <nav style = {navbarStyle}>
        {/* Logo */}
        <div> 
            <img src={logo} alt = "Foodlens Logo" style ={logoStyle} />
            {/* <span style={{ fontWeight: "bold", fontSize: "1.2rem" }}>FoodLens</span> */}
        </div>

        {/*Navigation Links */}
        <ul style= {navLinkStyle}>
            <li style = { navItemStyle}>
                <a 
                    href = "#home" 
                    style={{
                    ...LinkStyle,
                    ...(hovered === "home" ? linkHoverStyle : {}),
                    }}
                    onMouseEnter={() => setHovered("home")}
                    onMouseLeave={() => setHovered(null)}
                >
                    Home
                </a>
            </li>
            <li style = { navItemStyle}>
                <a 
                    href="#about"
                    style={{
                    ...LinkStyle,
                    ...(hovered === "about" ? linkHoverStyle : {}),
                    }}
                    onMouseEnter={() => setHovered("about")}
                    onMouseLeave={() => setHovered(null)}
                >
                  About
                </a>  
            </li>
        </ul>
    </nav>
  );
};

export default Navbar;