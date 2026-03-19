s-- This script is designed to be safely re-run.
-- It drops tables in the correct order to avoid foreign key constraint errors.

-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS attendance_system_db;
USE attendance_system_db;

-- Drop dependent tables first to satisfy foreign key constraints
DROP TABLE IF EXISTS `otplogs`;
DROP TABLE IF EXISTS `attendance`;
DROP TABLE IF EXISTS `sessions`;

-- Drop primary tables last
DROP TABLE IF EXISTS `students`;
DROP TABLE IF EXISTS `faculty`;
DROP TABLE IF EXISTS `admins`;

-- --- Table Creation ---

-- Table for storing student details
-- SAP ID is the unique key for each student.
CREATE TABLE `students` (
  `student_id` INT AUTO_INCREMENT PRIMARY KEY,
  `sap_id` VARCHAR(50) NOT NULL UNIQUE,
  `full_name` VARCHAR(100) NOT NULL,
  `phone_number` VARCHAR(20),
  `student_embedding` TEXT NOT NULL, -- Stores the FaceNet embedding as a JSON string
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing administrator details
CREATE TABLE `admins` (
    `admin_id` INT AUTO_INCREMENT PRIMARY KEY,
    `username` VARCHAR(50) NOT NULL UNIQUE,
    `password` VARCHAR(255) NOT NULL, -- In a real app, this should be a hash
    `full_name` VARCHAR(100) NOT NULL
);

-- Table for storing faculty details
CREATE TABLE `faculty` (
    `faculty_id` INT AUTO_INCREMENT PRIMARY KEY,
    `username` VARCHAR(50) NOT NULL UNIQUE,
    `password` VARCHAR(255) NOT NULL, -- In a real app, this should be a hash
    `full_name` VARCHAR(100) NOT NULL
);

-- Table for class/course sessions started by faculty
CREATE TABLE `sessions` (
    `session_id` INT AUTO_INCREMENT PRIMARY KEY,
    `faculty_id` INT,
    `course_name` VARCHAR(100) NOT NULL,
    `start_time` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `end_time` TIMESTAMP NULL,
    FOREIGN KEY (`faculty_id`) REFERENCES `faculty`(`faculty_id`) ON DELETE SET NULL
);

-- Table for logging attendance for each session
CREATE TABLE `attendance` (
    `attendance_id` INT AUTO_INCREMENT PRIMARY KEY,
    `session_id` INT,
    `student_id` INT,
    `status` ENUM('present', 'absent', 'flagged') NOT NULL,
    `timestamp` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (`session_id`) REFERENCES `sessions`(`session_id`) ON DELETE CASCADE,
    FOREIGN KEY (`student_id`) REFERENCES `students`(`student_id`) ON DELETE CASCADE
);

-- Table for OTP logs when face recognition confidence is low
CREATE TABLE `otplogs` (
    `otp_id` INT AUTO_INCREMENT PRIMARY KEY,
    `student_id` INT,
    `otp_code` VARCHAR(10) NOT NULL,
    `expiry_time` TIMESTAMP NOT NULL,
    `is_verified` BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (`student_id`) REFERENCES `students`(`student_id`) ON DELETE CASCADE
);

-- --- Default Admin User ---
-- Insert a default admin so you can log in later.
-- Username: admin, Password: password123
INSERT INTO `admins` (username, password, full_name)
VALUES ('admin', 'password123', 'Default Admin');

INSERT INTO faculty (username, password, full_name)
VALUES ('teacher', 'password123', 'Default Teacher');

SELECT 'Database schema created successfully' AS status;

