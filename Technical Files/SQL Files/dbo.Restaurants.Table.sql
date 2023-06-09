USE [ExploreCan]
GO
/****** Object:  Table [dbo].[Restaurants]    Script Date: 30-03-2023 12:47:12 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Restaurants](
	[Restaurant_Name] [nvarchar](50) NOT NULL,
	[Cuisine_Type] [nvarchar](50) NOT NULL,
	[Address] [nvarchar](50) NOT NULL,
	[City] [nvarchar](50) NOT NULL,
	[Province_State] [nvarchar](50) NOT NULL,
	[Country] [nvarchar](50) NOT NULL,
	[Postal_Zip_Code] [nvarchar](50) NOT NULL,
	[Phone_Number] [nvarchar](50) NOT NULL,
	[Price_Range_per_person] [nvarchar](50) NOT NULL,
	[Average_Rating_out_of_5] [float] NOT NULL,
	[Number_of_Reviews] [int] NOT NULL
) ON [PRIMARY]
GO
