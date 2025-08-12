// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./*.html"],
  theme: {
    extend: {
      colors: {
        // You could define custom colors if Tailwind's defaults aren't precise enough
        // 'primary-yellow': '#f8c200',
        // 'dark-grey-custom': '#333333',
        // 'light-grey-text-custom': '#cccccc',
      },
      fontFamily: {
        sans: ['Arial', 'sans-serif'], // Custom font family
      },
    },
  },
  plugins: [],
}