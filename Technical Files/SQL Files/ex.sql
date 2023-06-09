USE [master]
GO

/****** Object:  Database [ExploreCan]    Script Date: 30-03-2023 12:48:30 PM ******/
CREATE DATABASE [ExploreCan]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'ExploreCan', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ExploreCan.mdf' , SIZE = 8192KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'ExploreCan_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ExploreCan_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT
GO

IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [ExploreCan].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO

ALTER DATABASE [ExploreCan] SET ANSI_NULL_DEFAULT OFF 
GO

ALTER DATABASE [ExploreCan] SET ANSI_NULLS OFF 
GO

ALTER DATABASE [ExploreCan] SET ANSI_PADDING OFF 
GO

ALTER DATABASE [ExploreCan] SET ANSI_WARNINGS OFF 
GO

ALTER DATABASE [ExploreCan] SET ARITHABORT OFF 
GO

ALTER DATABASE [ExploreCan] SET AUTO_CLOSE OFF 
GO

ALTER DATABASE [ExploreCan] SET AUTO_SHRINK OFF 
GO

ALTER DATABASE [ExploreCan] SET AUTO_UPDATE_STATISTICS ON 
GO

ALTER DATABASE [ExploreCan] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO

ALTER DATABASE [ExploreCan] SET CURSOR_DEFAULT  GLOBAL 
GO

ALTER DATABASE [ExploreCan] SET CONCAT_NULL_YIELDS_NULL OFF 
GO

ALTER DATABASE [ExploreCan] SET NUMERIC_ROUNDABORT OFF 
GO

ALTER DATABASE [ExploreCan] SET QUOTED_IDENTIFIER OFF 
GO

ALTER DATABASE [ExploreCan] SET RECURSIVE_TRIGGERS OFF 
GO

ALTER DATABASE [ExploreCan] SET  DISABLE_BROKER 
GO

ALTER DATABASE [ExploreCan] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO

ALTER DATABASE [ExploreCan] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO

ALTER DATABASE [ExploreCan] SET TRUSTWORTHY OFF 
GO

ALTER DATABASE [ExploreCan] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO

ALTER DATABASE [ExploreCan] SET PARAMETERIZATION SIMPLE 
GO

ALTER DATABASE [ExploreCan] SET READ_COMMITTED_SNAPSHOT OFF 
GO

ALTER DATABASE [ExploreCan] SET HONOR_BROKER_PRIORITY OFF 
GO

ALTER DATABASE [ExploreCan] SET RECOVERY SIMPLE 
GO

ALTER DATABASE [ExploreCan] SET  MULTI_USER 
GO

ALTER DATABASE [ExploreCan] SET PAGE_VERIFY CHECKSUM  
GO

ALTER DATABASE [ExploreCan] SET DB_CHAINING OFF 
GO

ALTER DATABASE [ExploreCan] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO

ALTER DATABASE [ExploreCan] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO

ALTER DATABASE [ExploreCan] SET DELAYED_DURABILITY = DISABLED 
GO

ALTER DATABASE [ExploreCan] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO

ALTER DATABASE [ExploreCan] SET QUERY_STORE = OFF
GO

ALTER DATABASE [ExploreCan] SET  READ_WRITE 
GO

