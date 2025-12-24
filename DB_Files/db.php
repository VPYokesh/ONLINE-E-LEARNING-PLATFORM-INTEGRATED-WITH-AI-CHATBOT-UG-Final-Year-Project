<?php
$host = "localhost";
$user = "root";
$database = "elearn";
$password = "";

if ($conn = new mysqli($host, $user, $password, $database)) {
} else {
    echo 'Database Error';
    exit;
}
?>