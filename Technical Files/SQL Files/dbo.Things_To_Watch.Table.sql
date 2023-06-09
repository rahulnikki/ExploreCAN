USE [ExploreCan]
GO
/****** Object:  Table [dbo].[Things_To_Watch]    Script Date: 30-03-2023 12:47:12 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Things_To_Watch](
	[longitude] [nvarchar](50) NULL,
	[latitude] [nvarchar](100) NULL,
	[code] [nvarchar](50) NULL,
	[name] [nvarchar](100) NULL,
	[type] [nvarchar](50) NULL,
	[phone] [nvarchar](50) NULL,
	[dates_open] [nvarchar](50) NULL,
	[amenities] [nvarchar](50) NULL,
	[state] [nvarchar](50) NULL,
	[State_Name] [nvarchar](50) NULL,
	[city] [nvarchar](50) NULL
) ON [PRIMARY]
GO
