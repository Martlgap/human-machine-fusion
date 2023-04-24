/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./node_modules/flowbite/**/*.js",
    "./node_modules/tw-elements/dist/js/**/*.js",
    "./templates/**/*.{html,js}", 
    "./static/src/**/*.js"],
  plugins: [
    // require("flowbite/plugin"),
    // require("tw-elements/dist/plugin")
  ],
};
