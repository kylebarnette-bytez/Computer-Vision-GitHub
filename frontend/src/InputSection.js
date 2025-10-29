import { useRef, useState } from "react";

function InputSection({ onUpload, onType, onCamera, loading, result })
{

    const fileInputRef = useRef();
    const [foodName, setFoodName] = useState("");

    const sectionStyle ={
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "80vh",
        textAlign: "center",
        backgroundColor: "#171c34ff",       
        width: "100%",            
        boxSizing: "border-box",
        padding: "4rem",
    };

    const headingStyle = {
        fontSize: "2rem",
        color: "#88b2c8ff",
        marginTop : "2rem",
        marginBottom: "2rem",
        maxWidth: "600px"
    };

    const buttonConatinerStyle ={
        display: "flex",
        gap: "1rem",
        justifyContent: "center",
        flexWrap: "wrap",
        marginBotton: "1rem",
    };

    const buttonStyle = {
        backgroundColor: "#1e628a",
        color: "white",
        border: "none",
        borderRadius: "8px",
        padding: "0.75rem 2.5rem",
        fontSize: "1rem",
        cursor: "pointer",
        transition: "background-color 0.3s ease",
        margin: "0 0.5rem"
  };
  const resultBoxStyle = {
    width: "200px",
    height: "200px",
    backgroundColor: "#347e97ff",
    color: "white",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    marginTop: "3rem",
    alignItems: "center",
    marginBottom: "2rem",
    boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
    fontWeight: "bold",

  };


  const handleHover = (e, hover) => {
    e.target.style.backgroundColor = hover ? "#144d6b" : "#1e628a";
  };
  return (
    <section style={sectionStyle}>
      <h1 style={headingStyle}>
          Snap it. Know it. Eat smarter.
        </h1>

      <div style={{buttonConatinerStyle }}>
        <button
          style={buttonStyle}
          disabled={loading}
          onClick={() => fileInputRef.current.click()}
          onMouseEnter={(e) => handleHover(e, true)}
          onMouseLeave={(e) => handleHover(e, false)}
        >
          Upload Image
        </button>
        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => e.target.files[0] && onUpload(e.target.files[0])}
        />

         {/* Controlled input for food name */}
        <input
          type="text"
          placeholder="Type food name"
          value={foodName}
          onChange={(e) => setFoodName(e.target.value)}
          onKeyDown={(e) => {
        if (e.key === "Enter" && foodName.trim() !== "") {
                    onType(foodName);
        }
        }}
          style={{
            padding: "0.75rem 2rem",
            borderRadius: "8px",
            border: "1px solid #1e628a",
            fontSize: "1rem",
            minWidth: "200px",
          }}
        />
        <button
          style={buttonStyle}
          disabled={loading}
          onClick={onCamera}
          onMouseEnter={(e) => handleHover(e, true)}
          onMouseLeave={(e) => handleHover(e, false)}
        >
          Take Picture
        </button>
      </div>

      {/* Result box below buttons */}
      {result && (
        <div style={resultBoxStyle}>
          <p>Name: {result.predicted_food}</p>
          <p>Calories: {result.calories}</p>
          <p>Price: ${result.price}</p>
            </div>
      )}
    </section>
  );
};
export default InputSection;
